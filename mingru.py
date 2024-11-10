import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .min_ops import log_g, parallel_scan_log


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
        return h[:, -x.size(1) :, :]
