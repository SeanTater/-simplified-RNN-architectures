import torch
from torch import nn, optim
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from .lm import LanguageModel
from .pydantic_models import TrainConfig
from .tokenizer import tokenizer


class Trainer:
    """
    A class to handle the training and evaluation of a language model.

    Attributes:
        model (LanguageModel): The language model to train.
        config (TrainConfig): Configuration settings for training.
        optimizer (optim.Optimizer): Optimizer used for training the model.
        scaler (torch.GradScaler): Gradient scaler for mixed precision training.
        scheduler (LambdaLR): Learning rate scheduler.
    """

    model: LanguageModel
    config: TrainConfig
    optimizer: optim.Optimizer
    scaler: torch.GradScaler = torch.GradScaler()
    scheduler: LambdaLR
    device = torch.device

    def __init__(self, config: TrainConfig, model: LanguageModel):
        """
        Initializes the Trainer class.

        Args:
            config (TrainConfig): Configuration settings for training.
            model (LanguageModel): The language model to be trained.
        """
        self.device = torch.device(config.device)
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
        """
        Trains the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader object containing training data.

        Returns:
            float: Average loss over the entire training dataset.
        """
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
        """
        Evaluates the model on a validation dataset.

        Args:
            val_loader (DataLoader): DataLoader object containing validation data.

        Returns:
            float: Average loss over the entire validation dataset.
        """
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
        """
        Saves the model's state dictionary to a specified file.

        Args:
            path (str): File path where the model will be saved.
        """
        torch.save(self.model.state_dict(), path)
