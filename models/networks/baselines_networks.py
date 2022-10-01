"""
Adapted from https://github.com/emited/VariationalRecurrentNeuralNetwork

For the Trajectron++ baseline
It assumes a Categorical latent variable z. For every possible z (represented with one-hot encoding), the parameters
of a Gaussian distribution (assumed multi-dimensional and uncorrelated) are predicted for every time-step.

We support only two of the four sampling modes in the Trajectron++ paper: the "most likely" (mode), and the "full",
where z and y are sampled sequentially (regular mode).
"""

import torch
import torch.nn as nn

import distances
from models.networks.implicit import ResnetFC

__all__ = ['TrajEncoder', 'EncoderVRNN', 'EncoderTrajectron', 'TrajectronDecoder']


class TrajEncoder(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.phi_x = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())

    def forward(self, x):
        # features for x
        x_feat = self.phi_x(x)
        return x_feat


class EncoderVRNN(nn.Module):
    def __init__(self, hidden_size, feature_size, num_layers, num_latent_params, in_rnn_size=None, sample_fn=None,
                 **kwargs):
        if in_rnn_size is None:
            in_rnn_size = 2 * hidden_size
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_latent_params = num_latent_params
        self.sample_fn = sample_fn

        # Feature-extracting transformations
        self.phi_z = nn.Sequential(
            nn.Linear(feature_size, feature_size),
            nn.ReLU())

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.enc_mean = nn.Linear(hidden_size, feature_size)

        if num_latent_params == 2:
            self.enc_logvar = nn.Sequential(
                nn.Linear(hidden_size, feature_size),
                nn.Softplus())
        else:
            self.enc_logvar = None

        # Prior
        self.prior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        self.prior_mean = nn.Linear(hidden_size, feature_size)
        if num_latent_params == 2:
            self.prior_logvar = nn.Sequential(
                nn.Linear(hidden_size, feature_size),
                nn.Softplus())
        else:
            self.prior_logvar = None

        # Recurrence
        self.rnn = nn.GRU(in_rnn_size, hidden_size, num_layers, bias=True)

    def _forward(self, x_t, h):
        # x_t is already a feature representation for x_t
        x_feat = x_t

        if h is None:
            h = torch.zeros(self.num_layers, x_t.shape[0], self.hidden_size, device=x_t.device)

        # encoder
        if len(h.shape) - 1 > len(x_feat.shape):
            input_encoder = torch.cat([x_feat[:, None].expand(x_feat.shape[0], h.shape[2], x_feat.shape[-1]),
                                       h[-1]], -1)
        else:
            input_encoder = torch.cat([x_feat, h[-1]], 1)
        enc_t = self.enc(input_encoder)
        z_mean_t = self.enc_mean(enc_t)
        if self.num_latent_params == 2:
            z_logvar_t = self.enc_logvar(enc_t)
        else:
            z_logvar_t = None
        z_distr = distances.from_params(self.num_latent_params, z_mean_t, z_logvar_t)

        # prior
        prior_t = self.prior(h[-1])
        prior_mean_t = self.prior_mean(prior_t)
        if self.num_latent_params == 2:
            prior_logvar_t = self.prior_logvar(prior_t)
        else:
            prior_logvar_t = None
        prior_distr = distances.from_params(self.num_latent_params, prior_mean_t, prior_logvar_t)

        return z_distr, prior_distr, x_feat

    def forward(self, x_t, h, **kwargs):
        z_distr, prior_distr, x_feat = self._forward(x_t, h)

        # sampling and reparameterization
        sample_mode = None
        # After the first step, at evaluation and test time always sample the mode
        if not kwargs['training'] and not kwargs['first_step']:
            sample_mode = True
        z_t = self.sample_fn(z_distr, sample_mode=sample_mode).squeeze(1)
        z_feat = self.phi_z(z_t)

        if len(z_feat.shape) > len(x_feat.shape):  # Probably multiple noise samples per each element in the batch
            input_rnn = torch.cat([x_feat[:, None].expand(x_feat.shape[0], z_feat.shape[1], x_feat.shape[-1]),
                                   z_feat], -1)
            input_rnn = input_rnn.view(-1, input_rnn.shape[-1])
            if h is not None:
                h = h.view(h.shape[0], -1, h.shape[-1])
        else:
            input_rnn = torch.cat([x_feat, z_feat], 1)
        _, h_new = self.rnn(input_rnn.unsqueeze(0), h)

        if len(z_feat.shape) > len(x_feat.shape):
            h_new = h_new.view(h_new.shape[0], *z_feat.shape[:2], h_new.shape[-1])

        return z_distr, z_feat, prior_distr, h_new


class EncoderTrajectron(EncoderVRNN):
    """
    This is not exactly the Trajectron++ architecture, but it keeps the main elements of the Trajectron.
    The e_x in the paper (which includes contextual information as well as past) here is the "h" (past information).
    """

    def __init__(self, hidden_size, feature_size, in_size, num_layers, num_latent_params, sample_fn, **kwargs):
        super().__init__(hidden_size, feature_size, num_layers, num_latent_params, in_rnn_size=hidden_size,
                         sample_fn=sample_fn)

    def forward(self, x_t, h, training=True, **kwargs):
        z_distr, prior_distr, x_feat = self._forward(x_t, h)

        # sampling and reparameterization
        z_t = self.sample_fn(z_distr, return_all=True, grad_through=training).squeeze(-3)
        z_feat = self.phi_z(z_t)

        # recurrence. Does not include z_feat. RNN is oblivious of randomness.
        _, h_new = self.rnn(x_feat.unsqueeze(0), h)

        return z_distr, z_feat, prior_distr, h_new


class TrajectronDecoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, feature_size, n_blocks, num_layers_rnn=2,
                 gaussian_sampling=False):
        """
        If gaussian_sampling, before the decoder, we sample from a Gaussian.
        """
        super().__init__()
        self.num_layers_rnn = num_layers_rnn
        self.hidden_size = hidden_size
        self.gaussian_sampling = gaussian_sampling

        output_size_gru = hidden_size if not gaussian_sampling else 2 * hidden_size
        self.rnn = nn.GRU(hidden_size + feature_size, output_size_gru, num_layers_rnn, bias=True)
        self.decoder = ResnetFC(in_size=0, hidden_size=hidden_size, out_size=out_size, feature_size=hidden_size,
                                n_blocks=n_blocks)

    def forward(self, features_z, features_context, gt_x):

        y = []
        batch_size = features_z.shape[0] * features_z.shape[1]
        h = torch.zeros(self.num_layers_rnn, batch_size, self.hidden_size, device=gt_x.device)
        for t in range(gt_x.shape[1]):  # Temporal steps in the future

            inp = torch.cat([features_z, features_context], -1)
            inp = inp.view(-1, inp.shape[-1])
            output_rnn, h = self.rnn(inp.unsqueeze(0), h)

            if self.gaussian_sampling:
                mean, logvar = distances.get_params(output_rnn, 2)
                std = (logvar / 2).exp()
                output_rnn = std[..., None, :] * torch.randn(std.shape, device=std.device) + mean[..., None, :]

            y_t = self.decoder(output_rnn)
            y_t = y_t.view(features_z.shape[0], features_z.shape[1], -1)
            y.append(y_t)

        return y
