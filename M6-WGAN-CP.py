# NOTE: This file is intentionally a copy of `D_GAN.py` for manual switching.
# Variant target: WGAN-CP (weight clipping in training script).

import math

import torch
import torch.nn as nn


def weight_clipping_(model: nn.Module, clip_value: float):
    """In-place weight clipping for WGAN-CP."""
    if clip_value <= 0:
        raise ValueError("clip_value must be > 0")
    with torch.no_grad():
        for p in model.parameters():
            p.clamp_(-clip_value, clip_value)


def sample_z(batch_size: int, latent_dim: int, device):
    return torch.randn(batch_size, latent_dim, device=device)


class WGANGenerator1D(nn.Module):
    """1D Generator for vibration signals.

    WGAN-CP is trained in a centered [-1, 1] signal space by default.  The
    generated samples are min-max normalized again when written to .mat files.
    """

    def __init__(self, latent_dim: int = 128, out_length: int = 1024, base_channels: int = 128,
                 output_activation: str = "tanh"):
        super().__init__()
        if out_length % 16 != 0:
            raise ValueError("out_length must be a multiple of 16")
        ratio = out_length // 16
        if ratio & (ratio - 1) != 0:
            raise ValueError("out_length must be 16 * 2^k")
        self.latent_dim = latent_dim
        self.out_length = out_length
        self.output_activation = output_activation

        self.fc = nn.Sequential(nn.Linear(latent_dim, base_channels * 8 * 16), nn.LeakyReLU(0.2, inplace=True))

        num_ups = int(math.log2(ratio))

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.GroupNorm(8, out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )

        channels = [base_channels * 8]
        min_ch = max(8, base_channels // 8)
        for _ in range(num_ups):
            channels.append(max(min_ch, channels[-1] // 2))

        blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(up_block(in_ch, out_ch))
        self.up = nn.Sequential(*blocks)

        out_layers = [nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1)]
        if output_activation == "tanh":
            out_layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            out_layers.append(nn.Sigmoid())
        elif output_activation in (None, "none", "linear"):
            pass
        else:
            raise ValueError("output_activation must be 'tanh', 'sigmoid', or 'linear'")
        self.out = nn.Sequential(*out_layers)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.shape[0], -1, 16)
        x = self.up(x)
        return self.out(x)


class WGANCritic1D(nn.Module):
    """1D Critic (no sigmoid)."""

    def __init__(self, in_length: int = 1024, base_channels: int = 64):
        super().__init__()
        if in_length % 64 != 0:
            raise ValueError("in_length must be divisible by 64 (6 downsampling blocks)")
        self.in_length = int(in_length)

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            down_block(1, base_channels),
            down_block(base_channels, base_channels * 2),
            down_block(base_channels * 2, base_channels * 4),
            down_block(base_channels * 4, base_channels * 8),
            down_block(base_channels * 8, base_channels * 8),
            down_block(base_channels * 8, base_channels * 8),
        )
        final_len = self.in_length // 64
        self.head = nn.Linear(base_channels * 8 * final_len, 1)

    def features(self, x):
        h = self.net(x)
        return h.reshape(x.shape[0], -1)

    def forward(self, x):
        return self.head(self.features(x))


class VanillaDiscriminator1D(nn.Module):
    """1D Discriminator for vanilla GAN (returns logits, no sigmoid)."""

    def __init__(self, in_length: int = 1024, base_channels: int = 64):
        super().__init__()
        if in_length % 64 != 0:
            raise ValueError("in_length must be divisible by 64 (6 downsampling blocks)")
        self.in_length = int(in_length)

        def down_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.net = nn.Sequential(
            down_block(1, base_channels),
            down_block(base_channels, base_channels * 2),
            down_block(base_channels * 2, base_channels * 4),
            down_block(base_channels * 4, base_channels * 8),
            down_block(base_channels * 8, base_channels * 8),
            down_block(base_channels * 8, base_channels * 8),
        )
        final_len = self.in_length // 64
        self.head = nn.Linear(base_channels * 8 * final_len, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.reshape(x.shape[0], -1)
        return self.head(h)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, device, lambda_gp: float = 10.0):
    """WGAN-GP gradient penalty."""
    batch_size = real.shape[0]
    eps = torch.rand(batch_size, 1, 1, device=device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    d_hat = critic(x_hat)
    grads = torch.autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads = grads.view(batch_size, -1)
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp
