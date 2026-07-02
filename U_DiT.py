import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['UDiT1D', 'UNet1D']


def sinusoidal_positional_embedding(length, dim, device):
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    half_dim = dim // 2
    div_term = torch.exp(
        -math.log(10000.0) * torch.arange(half_dim, device=device, dtype=torch.float32) / max(1, half_dim)
    )
    emb = torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)
    if emb.shape[1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[1]))
    return emb


def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / max(1, half)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class AdaLayerNormDiTBlock1D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        x_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


class DownsampleTokens1D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class UpsampleFuseTokens1D(nn.Module):
    def __init__(self, in_dim, skip_dim, out_dim):
        super().__init__()
        self.fuse = nn.Conv1d(in_dim + skip_dim, out_dim, kernel_size=1)

    def forward(self, x, skip):
        x = x.transpose(1, 2)
        skip_cf = skip.transpose(1, 2)
        x = F.interpolate(x, size=skip_cf.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip_cf], dim=1)
        x = self.fuse(x)
        return x.transpose(1, 2)


class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return modulate(self.norm_final(x), shift, scale)


class UDiT1D(nn.Module):
    def __init__(
        self,
        c_in=1,
        c_out=1,
        time_dim=256,
        device="cuda",
        patch_size=8,
        base_dim=128,
        depths=(2, 2, 2),
        num_heads=(4, 8, 8),
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.device = device
        self.patch_size = patch_size
        self.time_dim = time_dim

        dim1 = base_dim
        dim2 = base_dim * 2
        dim3 = base_dim * 4

        self.patch_embed = nn.Conv1d(c_in, dim1, kernel_size=patch_size, stride=patch_size)
        self.t_embedder = TimestepEmbedder(time_dim)
        self.t_to_dim1 = nn.Linear(time_dim, dim1)
        self.t_to_dim2 = nn.Linear(time_dim, dim2)
        self.t_to_dim3 = nn.Linear(time_dim, dim3)

        self.enc1 = nn.ModuleList([
            AdaLayerNormDiTBlock1D(dim=dim1, num_heads=num_heads[0], mlp_ratio=mlp_ratio)
            for _ in range(depths[0])
        ])
        self.down1 = DownsampleTokens1D(dim1, dim2)

        self.enc2 = nn.ModuleList([
            AdaLayerNormDiTBlock1D(dim=dim2, num_heads=num_heads[1], mlp_ratio=mlp_ratio)
            for _ in range(depths[1])
        ])
        self.down2 = DownsampleTokens1D(dim2, dim3)

        self.mid = nn.ModuleList([
            AdaLayerNormDiTBlock1D(dim=dim3, num_heads=num_heads[2], mlp_ratio=mlp_ratio)
            for _ in range(depths[2])
        ])

        self.up2 = UpsampleFuseTokens1D(dim3, dim2, dim2)
        self.dec2 = nn.ModuleList([
            AdaLayerNormDiTBlock1D(dim=dim2, num_heads=num_heads[1], mlp_ratio=mlp_ratio)
            for _ in range(depths[1])
        ])

        self.up1 = UpsampleFuseTokens1D(dim2, dim1, dim1)
        self.dec1 = nn.ModuleList([
            AdaLayerNormDiTBlock1D(dim=dim1, num_heads=num_heads[0], mlp_ratio=mlp_ratio)
            for _ in range(depths[0])
        ])

        self.final_layer = FinalLayer1D(dim1)
        self.patch_unembed = nn.ConvTranspose1d(dim1, c_out, kernel_size=patch_size, stride=patch_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        for module in self.modules():
            if isinstance(module, AdaLayerNormDiTBlock1D):
                nn.init.constant_(module.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(module.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def _format_t(self, x, t):
        if not t.is_floating_point():
            t = t.float()
        if t.dim() == 0:
            t = t[None]
        if t.dim() > 1:
            t = t.view(t.shape[0], -1)[:, 0]
        t = t.to(x.device)
        return t

    def forward(self, x, t):
        b, _, l = x.shape
        t = self._format_t(x, t)

        padded_len = math.ceil(l / self.patch_size) * self.patch_size
        if padded_len != l:
            x = F.pad(x, (0, padded_len - l))

        tokens = self.patch_embed(x).transpose(1, 2)
        pos = sinusoidal_positional_embedding(tokens.shape[1], tokens.shape[2], tokens.device)
        tokens = tokens + pos.unsqueeze(0)

        t_emb = self.t_embedder(t)
        c1 = self.t_to_dim1(t_emb)
        c2 = self.t_to_dim2(t_emb)
        c3 = self.t_to_dim3(t_emb)

        x1 = tokens
        for blk in self.enc1:
            x1 = blk(x1, c1)

        x2 = self.down1(x1)
        for blk in self.enc2:
            x2 = blk(x2, c2)

        x3 = self.down2(x2)
        for blk in self.mid:
            x3 = blk(x3, c3)

        x = self.up2(x3, x2)
        for blk in self.dec2:
            x = blk(x, c2)

        x = self.up1(x, x1)
        for blk in self.dec1:
            x = blk(x, c1)

        x = self.final_layer(x, c1)
        out = self.patch_unembed(x.transpose(1, 2))

        if out.shape[-1] != l:
            out = out[..., :l]
        return out


class UNet1D(UDiT1D):
    """Drop-in alias to ease replacement from `U_Net.UNet1D` to `U_DiT.UNet1D`."""
    pass
