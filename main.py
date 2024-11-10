#!/usr/bin/env python
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pydantic_models import ModelConfig, Config
from .lm import LanguageModel
from .tokenizer import tokenizer
from .dataset import SinglefileDataset
from .trainer import Trainer


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
                output = self.model(tokens)
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
            f"Epoch {epoch+1}/{config.train.epochs},"
            f" Train Loss: {train_loss:.04f},"
            f" Val Loss: {val_loss:.04f},"
            f" LR: {trainer.scheduler.get_last_lr()[0]:.06f}"
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
            print(bot.chat("lizards are", max_length=10))
            print(bot.chat("python is", max_length=10))
            print(bot.chat("hello", max_length=10))

    print("Training completed.")
    if last_model_path is not None:
        print("Best model saved at", last_model_path)
        Chat.from_path(last_model_path).repl()


if __name__ == "__main__":
    main()
