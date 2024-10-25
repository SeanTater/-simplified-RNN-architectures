#!/usr/bin/env python
from pathlib import Path
from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from torch.amp import autocast
from argparse import ArgumentParser
from torch.optim.lr_scheduler import LambdaLR
import polars as pl
from pydantic_models import ModelConfig, TrainConfig, DatasetConfig, Config
import random

# Load the GPT-2 tokenizer
tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add a padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})


def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


def parallel_scan_log(log_coeffs, log_values):
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)


class MinGRU(nn.Module):
    linear_z: nn.Linear
    linear_h: nn.Linear
    def __init__(self, input_size: int, hidden_size: int):
        super(MinGRU, self).__init__()
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x: Tensor, h_0: Tensor):
        k: Tensor = self.linear_z(x)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_h_0 = log_g(h_0).unsqueeze(1)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(
            log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1)
        )
        return h[:, -x.size(1) :, :], 0

class GroupedMinGRU(nn.Module):
    linear_z: nn.Linear
    linear_h: nn.Linear
    n_groups: int
    def __init__(self, input_size: int, hidden_size: int, n_groups: int, entropy_weight: float = 0.01):
        super(GroupedMinGRU, self).__init__()
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)
        self.n_groups = n_groups
        self.entropy_weight = entropy_weight

    def forward(self, x: Tensor, h_0: Tensor):
        k: Tensor = self.linear_z(x)

        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)

        j = log_coeffs
        # Split log_z into n_groups groups, take their norms, softmax them, and then recombine them
        j = j.view(j.size(0), j.size(1), self.n_groups, j.size(2) // self.n_groups)
        j_norms = torch.linalg.vector_norm(j, dim=-1, keepdim=True)
        j_softmaxed = F.log_softmax(j_norms, dim=2)
        j_softmaxed_squeezed = j_softmaxed.squeeze(-1)
        softmax_entropy = -torch.sum(torch.exp(j_softmaxed_squeezed) * j_softmaxed_squeezed, dim=2).mean()

        print(j_norms.mean(axis=(0,1,3)))

        # if random.random() < 0.0001:
        #     breakpoint()
        
        j = (j / j_norms) * j_softmaxed
        j = j.view(j.size(0), j.size(1), -1)

        log_coeffs = j

        log_h_0 = log_g(h_0).unsqueeze(1)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(
            log_coeffs, torch.cat([log_h_0, log_z + log_tilde_h], dim=1)
        )
        return h[:, -x.size(1) :, :], #softmax_entropy * self.entropy_weight


class MinLSTM(nn.Module):
    linear_f: nn.Linear
    linear_i: nn.Linear
    linear_h: nn.Linear

    def __init__(self, input_size: int, hidden_size: int):
        super(MinLSTM, self).__init__()
        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x: Tensor, h_0: Tensor):
        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = log_g(h_0)
        log_tilde_h = log_g(self.linear_h(x))
        h = parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        return h


class LanguageModel(nn.Module):
    config: ModelConfig
    embedding: nn.Embedding
    rnn_type: Literal["mingru", "minlstm"]
    rnn_layers: nn.ModuleList
    layer_norm: nn.LayerNorm
    dropout: nn.Dropout
    fc: nn.Linear

    def __init__(self, config: ModelConfig, vocab_size: int):
        super(LanguageModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            vocab_size, config.embed_size, padding_idx=tokenizer.pad_token_id
        )
        self.rnn_type = config.rnn_type
        self.rnn_layers = nn.ModuleList()
        self.layer_norm = nn.LayerNorm(config.embed_size)
        self.dropout = nn.Dropout(config.dropout)
        for _ in range(config.num_layers):
            # if config.rnn_type == "minlstm":
            #     self.rnn_layers.append(MinLSTM(config.embed_size, config.hidden_size))
            # elif config.rnn_type == "mingru":
            self.rnn_layers.append(GroupedMinGRU(config.embed_size, config.hidden_size, 2, config.entropy_weight))
            # self.rnn_layers.append(MinGRU(config.embed_size, config.hidden_size))
        self.fc = nn.Linear(config.hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2)).to(x.device)
        entropy = 0
        for i, rnn in enumerate(self.rnn_layers):
            # Experiment: After the first layer, use dilated convolutions to increase the receptive field
            # Result: This is far, far too slow
            # if i > 1:
            #     x = self.convolution(x.permute(0, 2, 1)).permute(0, 2, 1)Maril
            #     # Each convolution should only be backward looking
            #     x = x[:, :-self.config.dilation, :]
            #     # Now we need to pad the sequence to the right
            #     x = F.pad(x, (0, 0, 0, self.config.dilation, 0, 0))

            # Experiment: After the first layer, use mean pooling to increase the receptive field
            # Result: this is fast but we can only look back 1 token, we want to look exponentially back with each layer
            # if i > 1:
            #     # Instead of using dilated convolutions, use mean pooling to increase the receptive field
            #     x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=2, stride=1, padding=1).permute(0, 2, 1)
            #     # Each pooling operation should only be backward looking
            #     x = x[:, :-1, :]
            #     # It was already padded, we don't pad it again

            # Experiment: After the first layer, unfold, mean pool, and fold to increase the receptive field from way back
            # if i > 1:
            #     # Unfold the sequence, which introduces the dilation too
            #     x = x.unfold(dim=1, size=2, step=self.config.dilation)
            #     # Mean pool the sequence
            #     x = F.avg_pool1d(x.permute(0, 2, 1), kernel_size=2, stride=1, padding=1).permute(0, 2, 1)
            #     # Fold the sequence back
            #     x = self.fold(x.permute(0, 2, 1)).permute(0, 2, 1)
            x, layer_entropy = rnn(x, h)
            x = self.layer_norm(x)
            x = self.dropout(x)
            entropy += layer_entropy
        return self.fc(x), entropy


class SinglefileDataset(Dataset):
    frame: pl.DataFrame
    config: DatasetConfig

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.frame = pl.read_parquet(config.file_path)[:1000]
        tokens = pl.Series([], dtype=pl.List(pl.UInt16))
        for slc in tqdm(self.frame.iter_slices(100), desc="Tokenizing training data..", total=(self.frame.height + 99) // 100):
            slice_tokens = [ self.encode_one_doc(doc) for doc in slc["markdown"] ]
            tokens.append(pl.Series(slice_tokens, dtype=pl.List(pl.UInt16)))
        tokens.rechunk()
        self.frame = self.frame.with_columns(tokens=tokens)
    

    def encode_one_doc(self, doc: str):
        return tokenizer.encode(
            doc, add_special_tokens=False
        )


    def __len__(self):
        return min(self.config.cut_to, self.frame.height)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens: torch.Tensor = self.frame["tokens"][idx].to_torch()

        # Create input and target sequences
        input_ids = tokens[:-1].to(dtype=torch.long)
        target_ids = tokens[1:].to(dtype=torch.long)

        # Pad both ends with one less than the sequence length
        pad_length = self.config.seq_length - 1
        input_ids = F.pad(
            input_ids, (pad_length, pad_length), value=tokenizer.pad_token_id
        )
        target_ids = F.pad(
            target_ids, (pad_length, pad_length), value=tokenizer.pad_token_id
        )

        # Work out a sliding window step that spans most of the sequence
        # step = max(1, len(input_ids) // self.config.batch_size)

        # Unfold into sliding context windows
        X = input_ids.unfold(0, self.config.seq_length, 1)[
            : self.config.batch_size, :
        ]
        Y = target_ids.unfold(0, self.config.seq_length, 1)[
            : self.config.batch_size, :
        ]

        # Pick a random batch_size elements from the unfolded tensor
        start = torch.randint(0, X.size(0), (self.config.batch_size,))

        return X[start], Y[start]


class Trainer:
    model: LanguageModel
    config: TrainConfig
    optimizer: optim.Optimizer
    scaler: torch.GradScaler = torch.GradScaler()
    scheduler: LambdaLR
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, config: TrainConfig, model: LanguageModel):
        self.model = model.to(self.device)
        self.model.compile()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=config.learning_rate, fused=True
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
        self.config = config

        # Learning rate scheduler
        start_end_periods = config.epochs // 10
        warmup_rates = torch.linspace(0.01, 1, start_end_periods)
        middle_rates = torch.ones(config.epochs - start_end_periods * 2)
        cooldown_rates = torch.linspace(1, 0.01, start_end_periods + 1)
        rates = torch.cat([warmup_rates, middle_rates, cooldown_rates])
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: rates[epoch])

    def train(self, train_loader):
        with autocast(device_type=self.device.type):
            self.model.train()
            total_loss = 0
            total_entropy  = 0
            progressbar = tqdm(train_loader, desc="Training")
            max_grads_acc = self.config.accumulate_grad_batches
            for i, batch in enumerate(progressbar):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                output, entropy = self.model(x)
                real_loss = self.criterion(output.view(-1, output.size(-1)), y.reshape(-1))
                loss = real_loss + entropy
                total_entropy += entropy
                self.scaler.scale(loss / max_grads_acc).backward()
                if (i + 1) % max_grads_acc == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1e-6)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                total_loss += real_loss.item()
                progressbar.set_postfix({"real loss": f"{total_loss / (i+1):.04f}", "entropy": f"{total_entropy / (i+1):.04f}"})

            self.scheduler.step()
        return total_loss / len(train_loader)

    def evaluate(self, val_loader):
        with autocast(device_type=self.device.type):
            self.model.eval()
            total_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    output, entropy = self.model(x)
                    loss = self.criterion(
                        output.view(-1, output.size(-1)), y.reshape(-1)
                    )
                    total_loss += loss.item()
        return total_loss / len(val_loader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)


class Chat:
    """Continue a string of text using the trained model"""

    model: LanguageModel
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model: LanguageModel):
        self.model = model
        self.model.to(self.device)

    @staticmethod
    def from_path(model_path: Path):
        model = LanguageModel(ModelConfig(), vocab_size=tokenizer.total_vocab_size)
        model.load_state_dict(torch.load(model_path.open("rb"), weights_only=True))

        return Chat(model)

    def chat(self, text: str, max_length: int = 100) -> str:
        self.model.eval()
        with torch.no_grad():
            tokens = tokenizer.encode(
                text, add_special_tokens=False, return_tensors="pt"
            )
            tokens = tokens.to(self.device)
            for _ in range(max_length):
                output, _entropy = self.model(tokens)
                next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
                tokens = torch.cat([tokens, next_token], dim=1)
                if next_token == tokenizer.eos_token_id:
                    break
            return tokenizer.decode(tokens[0], skip_special_tokens=True)

    def repl(self):
        while True:
            text = input("You: ")
            if text == "exit" or not text:
                break
            response = self.chat(text)
            print("Bot:", response)


def main():
    parser = ArgumentParser(description="Train a MinGRU model on a single text file")
    parser.add_argument("config_file", type=str, help="Path to the configuration TOML")
    args = parser.parse_args()

    config = Config.from_file(Path(args.config_file))

    vocab_size = tokenizer.total_vocab_size

    dataset = SinglefileDataset(config.dataset)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)

    model = LanguageModel(config.model, vocab_size=vocab_size)
    trainer = Trainer(config.train, model)

    best_val_loss = float("inf")
    last_model_path: Path | None = None
    for epoch in range(config.train.epochs):
        train_loss = trainer.train(train_loader)
        val_loss = trainer.evaluate(val_loader)
        print(
            f"Epoch {epoch+1}/{config.train.epochs}, Train Loss: {train_loss:.04f}, Val Loss: {val_loss:.04f}, LR: {trainer.scheduler.get_last_lr()[0]:.06f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Remove the old model
            if last_model_path is not None:
                last_model_path.unlink()
            last_model_path = Path(
                f"data/models/{config.experiment.name}-{config.experiment.created_at}-ce{best_val_loss:.03f}.pth"
            )
            trainer.save_model(last_model_path)
            bot = Chat(model)
            # Print some examples of the bot's responses
            print(f"Lizards are {bot.chat('lizards are', max_length=10)}")
            print(f"Python is {bot.chat('python is', max_length=10)}")
            print(f"Hello, {bot.chat('hello', max_length=10)}")

    print("Training completed.")
    if last_model_path is not None:
        print("Best model saved at", last_model_path)
        Chat.from_path(last_model_path).repl()


if __name__ == "__main__":
    main()