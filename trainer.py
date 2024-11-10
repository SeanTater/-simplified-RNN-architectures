from .lm import LanguageModel
from .pydantic_models import TrainConfig
from torch import optim, nn

from torch.optim.lr_scheduler import LambdaLR
import torch
from .tokenizer import tokenizer

from torch.amp import autocast
from tqdm import tqdm


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
            progressbar = tqdm(train_loader, desc="Training")
            max_grads_acc = self.config.accumulate_grad_batches
            for i, batch in enumerate(progressbar):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output.view(-1, output.size(-1)), y.reshape(-1))
                self.scaler.scale(loss / max_grads_acc).backward()
                if (i + 1) % max_grads_acc == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1e-6
                    )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                total_loss += loss.item()

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
                    output = self.model(x)
                    loss = self.criterion(
                        output.view(-1, output.size(-1)), y.reshape(-1)
                    )
                    total_loss += loss.item()
        return total_loss / len(val_loader)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
