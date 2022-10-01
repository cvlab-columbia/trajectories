import numpy as np
import torch
import torch.nn as nn

__all__ = ['PositionalEncoder']


class PositionalEncoder(nn.Module):
    def __init__(self, strategy, hidden_size, num_powers=16, d_in=1):
        super().__init__()
        assert strategy in ['fourier', 'mlp', 'identity']
        self.strategy = strategy
        self.hidden_size = hidden_size
        self.num_powers = num_powers  # only for fourier case
        self.d_in = d_in  # size of coordinates. If only time, it is 1

        if strategy == 'identity':
            self.network = nn.Identity()
        elif strategy == 'fourier':
            input_size = d_in * (num_powers * 2 + 1)
            self.network = nn.Linear(input_size, hidden_size, bias=True)
        else:  # strategy == 'mlp':
            self.network = nn.Sequential(nn.Linear(d_in, hidden_size, bias=True), nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size, bias=True))

    def forward(self, points):
        """

        :param points: [B, ..., d_in] coordinates to embed. In our case they only have 1 dimension (time)
        :return: [B, ..., self.hidden_size]
        """
        if self.d_in == 1 and points.shape[-1] != 1:
            points = points.unsqueeze(-1)
        if self.strategy == 'fourier':
            points = fourier_encode(points, base_frequency=0.1, num_powers=self.num_powers)
        embedded_points = self.network(points)

        return embedded_points


def fourier_encode(points, base_frequency: float, num_powers: int):
    """
    :param points (..., d_in). In our case d_in=1 because it is only time.
    :param base_frequency: First frequency for sin and cos.
        NOTE: Because of periodicity, this value should be such that the extent (diameter) of the
        scene is never larger than 1 / base_frequency.
    :param num_powers: F = number of powers of 2 of the base frequency to use.
    :return (..., d_in*(F*2+1)) tensor with Fourier encoded coordinates.
    """
    result = []

    # Calculate and include all F powers of two.
    for p in range(num_powers):
        cur_freq = base_frequency * (2 ** p)
        omega = cur_freq * np.pi * 2.0
        sin = torch.sin(points * omega)  # (..., d_in).
        cos = torch.cos(points * omega)  # (..., d_in).
        result.append(sin)
        result.append(cos)

    # Include original coordinates as well to ensure no information is lost.
    result = torch.cat([points, *result], dim=-1)

    return result
