#!/usr/bin/env python

# Import necessary libraries and modules from PyTorch, argparse for command line arguments,
# and other custom modules from the project.
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import SinglefileDataset
from .lm import LanguageModel
from .pydantic_models import Config, ModelConfig
from .tokenizer import tokenizer
from .trainer import Trainer


class Chat:
    """
    A class to continue a string of text using the trained model.

    Attributes:
        model (LanguageModel): The language model used for generating text.
        device (torch.device): The device on which the model is loaded (CPU or GPU).
    """

    model: LanguageModel
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, model: LanguageModel):
        """
        Initializes the Chat instance with a given language model and moves it to the available device.

        Args:
            model (LanguageModel): The pre-trained language model instance.
        """
        self.model = model
        self.model.to(self.device)

    @staticmethod
    def from_path(model_path: Path):
        """
        Static method to load a Chat instance from a saved model path.

        Args:
            model_path (Path): The file path where the trained model's state dictionary is stored.

        Returns:
            Chat: A new Chat instance initialized with the loaded model.
        """
        # Initialize the language model with default configuration and tokenizer's vocabulary size
        model = LanguageModel(ModelConfig(), vocab_size=tokenizer.total_vocab_size)

        # Load the model's state dictionary from the provided path
        model.load_state_dict(torch.load(model_path.open("rb"), weights_only=True))

        return Chat(model)

    def chat(self, text: str, max_length: int = 100) -> str:
        """
        Generates a continuation of the input text using the trained model.

        Args:
            text (str): The initial string of text to be continued.
            max_length (int, optional): The maximum length of the generated text. Defaults to 100.

        Returns:
            str: The continued text as generated by the model.
        """
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            # Encode the input text into tokens and move them to the device
            tokens = tokenizer.encode(
                text, add_special_tokens=False, return_tensors="pt"
            )
            tokens = tokens.to(self.device)

            # Iteratively generate next tokens until max_length is reached or EOS token is generated
            for _ in range(max_length):
                output = self.model(
                    tokens
                )  # Get model's output on the current set of tokens
                next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(
                    1
                )  # Select the most likely token
                tokens = torch.cat(
                    [tokens, next_token], dim=1
                )  # Append the generated token to the sequence

                # Break if the EOS (End Of Sentence) token is encountered
                if next_token.item() == tokenizer.eos_token_id:
                    break

            # Decode the final sequence of tokens into human-readable text and return it
            return tokenizer.decode(tokens[0], skip_special_tokens=True)

    def repl(self):
        """
        Read-Eval-Print Loop to interact with the chatbot in real-time.

        This method continuously takes input from the user, generates a response using the chat method,
        and prints the bot's reply. The loop exits when the user types 'exit' or provides an empty line.
        """
        while True:
            text = input("You: ")  # Take user input
            if text == "exit" or not text:  # Exit condition
                break
            response = self.chat(text)  # Generate bot's response
            print("Bot:", response)  # Print the bot's reply


def main():
    """
    Main function to train a MinGRU model on a single text file and interact with it.

    The script parses command line arguments, loads configuration settings from a TOML file,
    prepares datasets for training and validation, initializes the language model and trainer,
    performs the training process, evaluates the model, and optionally enters an interactive session
    with the trained chatbot.
    """
    # Setup argument parser to receive the path to the configuration file
    parser = ArgumentParser(description="Train a MinGRU model on a single text file")
    parser.add_argument("config_file", type=str, help="Path to the configuration TOML")
    args = parser.parse_args()

    # Load the configuration settings from the provided TOML file
    config = Config.from_file(Path(args.config_file))

    vocab_size = (
        tokenizer.total_vocab_size
    )  # Get the total vocabulary size from the tokenizer

    # Prepare datasets for training and validation
    dataset = SinglefileDataset(config.dataset)
    train_size = int(0.8 * len(dataset))  # Split 80% of data for training
    val_size = len(dataset) - train_size  # Remaining 20% for validation
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create DataLoader instances for batching during training and evaluation
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=None)
    val_loader = DataLoader(val_dataset, batch_size=None)

    # Initialize the language model with configuration and vocabulary size
    model = LanguageModel(config.model, vocab_size=vocab_size)

    # Create a Trainer instance to handle training and evaluation processes
    trainer = Trainer(config.train, model)

    best_val_loss = float("inf")  # Variable to track the best validation loss
    last_model_path: Path | None = (
        None  # Placeholder for the file path of the last saved best model
    )

    # Training loop over a specified number of epochs as defined in the configuration
    for epoch in range(config.train.epochs):
        train_loss = trainer.train(
            train_loader
        )  # Train the model on the training dataset
        val_loss = trainer.evaluate(
            val_loader
        )  # Evaluate the model on the validation dataset

        # Print the current epoch's training and validation losses alongside the learning rate
        print(
            f"Epoch {epoch+1}/{config.train.epochs},"
            f" Train Loss: {train_loss:.04f},"
            f" Val Loss: {val_loss:.04f},"
            f" LR: {trainer.scheduler.get_last_lr()[0]:.06f}"
        )

        # Save the model if it achieves a lower validation loss than previously recorded
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Remove the old model file if it exists to save disk space
            if last_model_path is not None:
                last_model_path.unlink()

            # Define the file path for the newly saved best model
            last_model_path = Path(
                f"data/models/{config.experiment.name}-{config.experiment.created_at}-ce{best_val_loss:.03f}.pth"
            )

            # Save the current state of the model to the defined path
            trainer.save_model(last_model_path)

            # Create a Chat instance with the best performing model for demonstration purposes
            bot = Chat(model)

            # Generate and print some sample responses from the chatbot to demonstrate its performance
            print(bot.chat("lizards are", max_length=10))
            print(bot.chat("python is", max_length=10))
            print(bot.chat("hello", max_length=10))

    print("Training completed.")  # Indicate that the training process has finished

    # If a best model was saved, inform the user of its location and enter an interactive session
    if last_model_path is not None:
        print("Best model saved at", last_model_path)
        Chat.from_path(last_model_path).repl()


if __name__ == "__main__":
    main()  # Execute the main function when the script is run directly
