from torch import nn, Tensor
import torch.nn.functional as F
from .min_ops import log_g, parallel_scan_log
import torch


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
