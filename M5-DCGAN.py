# NOTE: This file is intentionally separated for manual switching.
# Variant target: DCGAN-style baseline (BCE loss in training script).

import math

import torch
import torch.nn as nn


def sample_z(batch_size: int, latent_dim: int, device):
    return torch.randn(batch_size, latent_dim, device=device)


def weight_clipping_(model: nn.Module, clip_value: float):
    """In-place weight clipping helper (unused for DCGAN baseline)."""
    if clip_value <= 0:
        raise ValueError("clip_value must be > 0")
    with torch.no_grad():
        for p in model.parameters():
            p.clamp_(-clip_value, clip_value)


class WGANGenerator1D(nn.Module):
    """1D DCGAN-like generator for vibration signals.

    Output shape: (B,1,out_length). Uses Sigmoid to match dataset min-max scaling to [0,1].
    """

    def __init__(self, latent_dim: int = 128, out_length: int = 1024, base_channels: int = 512):
        super().__init__()
        if out_length % 16 != 0:
            raise ValueError("out_length must be a multiple of 16")
        ratio = out_length // 16
        if ratio & (ratio - 1) != 0:
            raise ValueError("out_length must be 16 * 2^k")
        self.latent_dim = latent_dim
        self.out_length = out_length

        # Project z -> (C,16)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * 16),
            nn.ReLU(True),
        )

        def deconv_block(in_ch: int, out_ch: int):
            return nn.Sequential(
                nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(True),
            )

        num_ups = int(math.log2(ratio))
        min_ch = max(8, base_channels // 32)

        channels = [base_channels]
        for _ in range(num_ups):
            channels.append(max(min_ch, channels[-1] // 2))

        blocks = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            blocks.append(deconv_block(in_ch, out_ch))
        blocks.append(nn.Conv1d(channels[-1], 1, kernel_size=3, padding=1, bias=True))
        blocks.append(nn.Sigmoid())
        self.net = nn.Sequential(*blocks)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.shape[0], -1, 16)
        return self.net(x)


class WGANCritic1D(nn.Module):
    """Critic backbone kept for API compatibility.

    Note: In proper comparisons, WGAN-CP/WGAN-GP should import their own module.
    """

    def __init__(self, in_length: int = 1024, base_channels: int = 64):
        super().__init__()
        if in_length % 64 != 0:
            raise ValueError("in_length must be divisible by 64 (6 downsampling blocks)")
        self.in_length = int(in_length)

        def down_block(in_ch: int, out_ch: int, use_bn: bool):
            layers = [nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            down_block(1, base_channels, use_bn=False),
            down_block(base_channels, base_channels * 2, use_bn=True),
            down_block(base_channels * 2, base_channels * 4, use_bn=True),
            down_block(base_channels * 4, base_channels * 8, use_bn=True),
            down_block(base_channels * 8, base_channels * 8, use_bn=True),
            down_block(base_channels * 8, base_channels * 8, use_bn=True),
        )
        final_len = self.in_length // 64
        self.head = nn.Linear(base_channels * 8 * final_len, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.reshape(x.shape[0], -1)
        return self.head(h)


class VanillaDiscriminator1D(nn.Module):
    """1D DCGAN-like discriminator (returns logits, no sigmoid)."""

    def __init__(self, in_length: int = 1024, base_channels: int = 64):
        super().__init__()
        if in_length % 64 != 0:
            raise ValueError("in_length must be divisible by 64 (6 downsampling blocks)")
        self.in_length = int(in_length)

        def down_block(in_ch: int, out_ch: int, use_bn: bool):
            layers = [nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16
        self.net = nn.Sequential(
            down_block(1, base_channels, use_bn=False),
            down_block(base_channels, base_channels * 2, use_bn=True),
            down_block(base_channels * 2, base_channels * 4, use_bn=True),
            down_block(base_channels * 4, base_channels * 8, use_bn=True),
            down_block(base_channels * 8, base_channels * 8, use_bn=True),
            down_block(base_channels * 8, base_channels * 8, use_bn=True),
        )
        final_len = self.in_length // 64
        self.head = nn.Linear(base_channels * 8 * final_len, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.reshape(x.shape[0], -1)
        return self.head(h)


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, device, lambda_gp: float = 10.0):
    """WGAN-GP gradient penalty (unused for DCGAN baseline)."""
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
