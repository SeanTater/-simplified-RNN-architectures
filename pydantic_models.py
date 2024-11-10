from pydantic import Field, FilePath, BaseModel
from datetime import datetime
from typing import Literal
import tomllib
from pathlib import Path


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    created_at: str = datetime.now().isoformat(timespec="seconds")


class ModelConfig(BaseModel):
    embed_size: int = Field(384 * 2, description="Size of the embedding layer")
    hidden_size: int = Field(384 * 2, description="Size of the hidden layer")
    num_layers: int = Field(3, description="Number of RNN layers")
    rnn_type: Literal["mingru", "minlstm"] = Field(
        "mingru", description="Type of RNN to use: 'mingru' or 'minlstm'"
    )
    dropout: float = Field(0.25, description="Dropout rate for the model")
    dilation: int = Field(2, description="Dilation factor for the model")
    entropy_weight: float = Field(0.1, description="Weight for the entropy loss")
    num_groups: int = Field(
        2, description="Number of groups used inside embedding vectors for the model"
    )


class TrainConfig(BaseModel):
    learning_rate: float = Field(1e-3, description="Learning rate for the optimizer")
    epochs: int = Field(50, description="Number of training epochs")
    accumulate_grad_batches: int = Field(
        1, description="Number of batches to accumulate gradients over"
    )


class DatasetConfig(BaseModel):
    file_path: FilePath = Field(..., description="Path to the dataset file")
    batch_size: int = Field(64, description="Batch size for training")
    seq_length: int = Field(100, description="Sequence length for training")
    byte_snippet_length: int = Field(
        2048, description="Length of byte snippets to read from the file"
    )
    cut_to: int = Field(1000, description="Cut the dataset to this many byte snippets")


class Config(BaseModel):
    model: ModelConfig
    train: TrainConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig

    @staticmethod
    def from_file(path: Path):
        return Config.model_validate(tomllib.loads(path.read_text()))
