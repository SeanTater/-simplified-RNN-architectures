import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .min_ops import log_g, parallel_scan_log


class MinGRU(nn.Module):
    linear_z: nn.Linear  # Linear transformation layer for computing k
    linear_h: (
        nn.Linear
    )  # Linear transformation layer for computing the transformed input

    def __init__(self, input_size: int, hidden_size: int):
        super(MinGRU, self).__init__()
        # Initialize the linear layers with the given input and hidden sizes
        self.linear_z = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def forward(self, x: Tensor, h_0: Tensor) -> Tensor:
        """
        Forward pass of the MinGRU module.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, seq_len, input_size).
            h_0 (Tensor): Initial hidden state tensor of shape (batch_size, hidden_size).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Step 1: Linear transformation to compute k
        # k represents a transformed version of the input x using linear_z
        k: Tensor = self.linear_z(x)

        # Step 2: Compute log_z, which is the logarithm of the complementary probabilities
        # -F.softplus(-k) computes the log of the complement of the sigmoid activation
        log_z = -F.softplus(-k)

        # Step 3: Compute log_coeffs, which are the logarithmic coefficients
        # -F.softplus(k) computes the logarithm of the negative probabilities associated with k
        log_coeffs = -F.softplus(k)

        # Step 4: Transform the initial hidden state h_0 using log_g and add a time step dimension
        # log_h_0 is the transformed initial hidden state, unsqueezed to have shape (batch_size, 1, hidden_size)
        log_h_0 = log_g(h_0).unsqueeze(1)

        # Step 5: Transform the input x using linear_h and then apply log_g
        # log_tilde_h is the transformed input values with logarithmic adjustments
        log_tilde_h = log_g(self.linear_h(x))

        # Step 6: Concatenate initial hidden state and the element-wise sum of log_z and log_tilde_h
        # The result is a tensor of shape (batch_size, seq_len + 1, hidden_size)
        concatenated_values = torch.cat([log_h_0, log_z + log_tilde_h], dim=1)

        # Step 7: Perform the parallel scan operation using the computed log_coeffs and concatenated values
        # This accumulates information across time steps while maintaining independence between different parts of the sequence
        h = parallel_scan_log(log_coeffs, concatenated_values)

        # Step 8: Slice the final output to return only the relevant part corresponding to the input sequence length
        # Extract the last x.size(1) time steps from the accumulated results
        return h[:, -x.size(1) :, :]
