"""
Adapted from https://github.com/cvlab-columbia/pcl-motion-forecast/blob/main/implicit.py
"""

import numpy as np
import torch
import torch.nn

__all__ = ['ResnetFC']


class Sine(torch.nn.Module):
    """
    https://arxiv.org/pdf/2006.09661.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30.0 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class Swish(torch.nn.Module):
    """
    https://arxiv.org/pdf/1710.05941.pdf
    """

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


def instantiate_activation_fn(activation_str):
    if activation_str == 'relu':
        return torch.nn.ReLU()
    elif activation_str == 'sine':
        return Sine()
    elif activation_str == 'swish':
        return Swish()
    else:
        raise ValueError('Unknown activation: ' + str(activation_str))


# Resnet Blocks
class ResnetBlockFC(torch.nn.Module):

    def __init__(self, d_in=64, d_hidden=256, d_out=64, activation='relu'):
        """
        Fully connected ResNet Block class. Taken from DVR code.
        :param d_in (int): input dimension.
        :param d_hidden (int): hidden dimension.
        :param d_out (int): output dimension.
        :param activation (str): relu / sine / swish.
        """
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out

        self.fc_0 = torch.nn.Linear(d_in, d_hidden, bias=True)
        self.fc_1 = torch.nn.Linear(d_hidden, d_out, bias=True)
        self.activation = instantiate_activation_fn(activation)

        if d_in == d_out:
            self.shortcut = None
        else:
            self.shortcut = torch.nn.Linear(d_in, d_out, bias=False)

    def forward(self, x):
        net = self.fc_0(self.activation(x))
        dx = self.fc_1(self.activation(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class ResnetFC(torch.nn.Module):
    """
    Regular continuous representation (CR) from pixelNeRF that also supports positional encoding
    and relu / sine / swish activation.
    """

    def __init__(self, in_size=1, hidden_size=256, out_size=64, feature_size=256, n_blocks=5, activation='relu',
                 freeze_all=False):
        """
        :param in_size (int): Input size (size of the coordinates)
        :param hidden_size (int): H = hidden dimension throughout network.
        :param out_size (int): G = output size of decoder. This is the size of the input space
        :param feature_size (int): D = latent size, added in each resnet block (0 = disable).
        :param n_blocks (int): number of resnet blocks.
        :param activation (str): relu / sine / swish.
        """
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.n_blocks = n_blocks

        self.lin_out = torch.nn.Linear(hidden_size, out_size, bias=True)

        self.blocks = torch.nn.ModuleList(
            [ResnetBlockFC(d_in=hidden_size, d_hidden=hidden_size, d_out=hidden_size,
                           activation=activation) for _ in range(n_blocks)])

        self.lin_z = torch.nn.ModuleList(
            [torch.nn.Linear(feature_size, hidden_size, bias=True) for _ in range(n_blocks)])

        if activation == 'sine':
            self.blocks.apply(sine_init)

        self.activation = instantiate_activation_fn(activation)

        if freeze_all:
            for name, parameter in self.named_parameters():
                parameter.requires_grad = False

    def forward(self, features, points=None, **kwargs):
        """
        :param points (B, ..., (T'), H) tensor. T' is the number of query points per element in the batch. If we query
            with more dimensions (...), we combine all of them into T.
        :param features (B, S, H). S is the number of latent space points sampled per element in the batch
        :return output: (B, T, S, G)
        """
        input_squeezed = False

        if points is None:
            assert self.in_size == 0, 'This has to be explicit to make sure it is intentional'
            x, initial_shape = None, None

        else:
            initial_shape = points.shape
            if len(points.shape) > 3:
                points = points.view(points.shape[0], -1, points.shape[-1])
                input_squeezed = True
            assert points.shape[0] == features.shape[0]
            x = points.unsqueeze(2)

        # Loop over all blocks.
        for blkid in range(self.n_blocks):
            # Add input features to the current representation in a residual manner.
            z = self.lin_z[blkid](features)  # (B, S, H).
            if points is None:
                x = z if x is None else x + z
            else:
                x = x + z.unsqueeze(1)  # (B, T, S, H).
            x = self.blocks[blkid](x)  # (B, T, S, H).
            # [or (B, H) if points is None, or (B, S, H) if points does not contain T]

        x = self.activation(x)  # (B, T, S, H).
        output = self.lin_out(x)  # (B, T, S, G) [or (B, S, G) if points is None]

        if input_squeezed:
            output = output.view([*initial_shape[:-1], *output.shape[-2:]])  # [B, ..., T', S, G]

        return output
