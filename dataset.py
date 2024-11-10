import polars as pl
from .pydantic_models import DatasetConfig
from tqdm import tqdm
from .tokenizer import tokenizer
import torch

from torch.utils.data import Dataset
from torch.nn import functional as F


class SinglefileDataset(Dataset):
    frame: pl.DataFrame
    config: DatasetConfig

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.frame = pl.read_parquet(config.file_path)
        tokens = pl.Series([], dtype=pl.List(pl.UInt16))
        for slc in tqdm(
            self.frame.iter_slices(100),
            desc="Tokenizing training data..",
            total=(self.frame.height + 99) // 100,
        ):
            slice_tokens = [self.encode_one_doc(doc) for doc in slc["markdown"]]
            tokens.append(pl.Series(slice_tokens, dtype=pl.List(pl.UInt16)))
        tokens.rechunk()
        self.frame = self.frame.with_columns(tokens=tokens)

    def encode_one_doc(self, doc: str):
        return tokenizer.encode(doc, add_special_tokens=False)

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
        X = input_ids.unfold(0, self.config.seq_length, 1)[: self.config.batch_size, :]
        Y = target_ids.unfold(0, self.config.seq_length, 1)[: self.config.batch_size, :]

        # Pick a random batch_size elements from the unfolded tensor
        start = torch.randint(0, X.size(0), (self.config.batch_size,))

        return X[start], Y[start]
