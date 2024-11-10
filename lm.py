from .pydantic_models import ModelConfig
from torch import nn
from typing import Literal
from .tokenizer import tokenizer
import torch
from .mingru import MinGRU


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
            # self.rnn_layers.append(GroupedMinGRU(config.embed_size, config.hidden_size, config.num_groups, config.entropy_weight))
            self.rnn_layers.append(MinGRU(config.embed_size, config.hidden_size))
        self.fc = nn.Linear(config.hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        h = torch.zeros(x.size(0), x.size(2)).to(x.device)
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
            x = rnn(x, h)
            x = self.layer_norm(x)
            x = self.dropout(x)
        return self.fc(x)
