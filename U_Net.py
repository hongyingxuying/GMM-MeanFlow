import torch
from torch._higher_order_ops import torchbind
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import math
import random
import numpy as np

use_cuda = torch.cuda.is_available()
if use_cuda:
    gpu = 0
device = torch.device("cuda:0" if use_cuda else "cpu")

class SelfAttention1D(nn.Module):
    def __init__(self, channels, length):
        super(SelfAttention1D, self).__init__()
        self.channels = channels
        self.length = length
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x_ln = self.ln(x)
        # We don't use attention weights; disabling them saves a lot of memory.
        attention_value, _ = self.mha(x_ln, x_ln, x_ln, need_weights=False)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.transpose(2, 1)

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, in_channels, residual=True),
            DoubleConv1D(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t_emb):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_emb)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv1D(in_channels, in_channels, residual=True),
            DoubleConv1D(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_emb)[:, :, None].repeat(1, 1, x.shape[-1])
        return x + emb

class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier features for continuous time embedding.
    - embed_dim must be even
    - scale controls frequency bandwidth (try 10~50; tune as needed)
    """
    def __init__(self, embed_dim=256, scale=30.0):
        super().__init__()
        assert embed_dim % 2 == 0, "time_dim must be even for sin/cos pair"
        # 固定随机频率（非可训练），用 register_buffer 保证移动 device 时同步
        W = torch.randn(embed_dim // 2) * scale
        self.register_buffer('W', W)

    def forward(self, t):
        # t: (B,) or (B,1), 值应归一化到 [0,1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        proj = t * self.W[None, :] * 2 * math.pi  # (B, embed_dim//2)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, embed_dim)

class UNet1D(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv1D(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention1D(128, 32)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention1D(256, 16)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.sa3 = SelfAttention1D(256, 8)

        self.bot1 = DoubleConv1D(256, 512)
        self.bot2 = DoubleConv1D(512, 512)
        self.bot3 = DoubleConv1D(512, 256)

        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.sa4 = SelfAttention1D(128, 16)
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.sa5 = SelfAttention1D(64, 32)
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.sa6 = SelfAttention1D(64, 64)
        self.outc = nn.Conv1d(64, c_out, kernel_size=1)#拼接两个分支后通道数变为128

        # --- continuous time embedding: Gaussian Fourier + small MLP ---
        self.time_proj = GaussianFourierProjection(time_dim, scale=30.0)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, x, t):
        """
        x: (B, C, L)
        t: scalar/float tensor, recommended normalized to [0,1], shape (B,) or (B,1)
        输出与原来一致： (B, c_out, L)
        """
        # ensure float and shape (B,1)
        if not t.is_floating_point():
            t = t.float()
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.to(x.device)

        # continuous time embedding
        t_emb = self.time_proj(t)           # (B, time_dim)
        t_emb = self.time_mlp(t_emb)       # (B, time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t_emb)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t_emb)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t_emb)
        x = self.sa4(x)
        x = self.up2(x, x2, t_emb)
        x = self.sa5(x)
        x = self.up3(x, x1, t_emb)
        x = self.sa6(x)

        output = self.outc(x)
        return output
