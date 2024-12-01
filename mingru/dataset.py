import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from .pydantic_models import DatasetConfig
from .tokenizer import tokenizer


class SinglefileDataset(Dataset):
    """
    A custom PyTorch dataset class for processing and tokenizing documents from a Parquet file.

    Attributes:
        frame (pl.DataFrame): Polars DataFrame containing the dataset.
        config (DatasetConfig): Configuration object specifying dataset parameters.
    """

    def __init__(self, config: DatasetConfig):
        """
        Initializes the SinglefileDataset by loading data and tokenizing documents.

        Args:
            config (DatasetConfig): Configuration object with dataset settings.
        """
        self.config = config
        # Load a subset of the Parquet file based on max_chunks configuration
        self.frame = pl.read_parquet(config.file_path).head(
            100 * self.config.max_chunks
        )

        # Initialize an empty Series to store tokenized documents
        tokens = pl.Series([], dtype=pl.List(pl.UInt16))

        # Tokenize each document in the dataset using a progress bar for better user feedback
        for slc in tqdm(
            self.frame.iter_slices(100),
            desc="Tokenizing training data..",
            total=(self.frame.height + 99) // 100,
        ):
            slice_tokens = [self.encode_one_doc(doc) for doc in slc["markdown"]]
            # Append tokenized slices to the main tokens Series
            tokens.append(pl.Series(slice_tokens, dtype=pl.List(pl.UInt16)))

        # Rechunk the Series to optimize memory layout and performance
        tokens.rechunk()
        # Add the tokenized columns back into the DataFrame
        self.frame = self.frame.with_columns(
            tokens=pl.Series(tokens, dtype=pl.List(pl.UInt16))
        )

    def encode_one_doc(self, doc: str) -> list[int]:
        """
        Tokenizes a single document string using the tokenizer.

        Args:
            doc (str): The document text to be tokenized.

        Returns:
            list[int]: A list of token IDs representing the document.
        """
        return tokenizer.encode(doc, add_special_tokens=False)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        This method returns either the number of documents or a cut-off value specified in the configuration,
        whichever is smaller.

        Returns:
            int: The effective length of the dataset.
        """
        return min(self.config.cut_to, self.frame.height)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single item from the dataset as a pair of input and target tensors.

        Args:
            idx (int): The index of the document to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input tensor and the corresponding
                                                target tensor for training purposes.
        """
        # Convert token list at the specified index to a PyTorch tensor
        tokens: torch.Tensor = self.frame["tokens"][idx].to_torch()

        # Create input and target sequences by shifting tokens by one position
        input_ids = tokens[:-1].to(dtype=torch.long)
        target_ids = tokens[1:].to(dtype=torch.long)

        # Calculate the padding length, which is the sequence length minus one
        pad_length = self.config.seq_length - 1

        # Pad both ends of the input and target sequences with padding token IDs
        input_ids = F.pad(
            input_ids, (pad_length, pad_length), value=tokenizer.pad_token_id
        )
        target_ids = F.pad(
            target_ids, (pad_length, pad_length), value=tokenizer.pad_token_id
        )

        # Create sliding context windows for the input and target sequences using unfold
        X = input_ids.unfold(0, self.config.seq_length, 1)[: self.config.batch_size, :]
        Y = target_ids.unfold(0, self.config.seq_length, 1)[: self.config.batch_size, :]

        # Randomly select a batch of context windows from the unfolded tensors
        start = torch.randint(0, X.size(0), (self.config.batch_size,))

        return X[start], Y[start]
