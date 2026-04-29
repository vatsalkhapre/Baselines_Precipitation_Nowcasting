"""
Incremental Ablation — Step 2: MLP + Gabor
No Wavelet, No conv_spectral.
GaborLayer + MLP in parallel at full resolution, fused via Conv3d.
Gabor residual added after fusion.
Lifting and Projection present.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from utils.utilspp import RandomScheduling


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size,
                              padding=kernel_size // 2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class TransformBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.skip   = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale,
                 alpha=1.0, beta=1.0, freq_multiplier=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu     = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma  = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        self.freq           = nn.Parameter(torch.rand(out_features))
        self.freq_multiplier = freq_multiplier

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.freq_multiplier * self.freq * self.linear(x)) * \
               torch.exp(-0.5 * D * self.gamma[None, :])


class TemporalBlock(nn.Module):
    """
    Step 2: MLP + Gabor in parallel at full resolution.
    Fused via Conv3d(2C -> C). Gabor residual added.
    No wavelet decomposition.
    """
    def __init__(self, t_in, t_out, dim,
                 weight_scale, alpha, beta, freq_multiplier,
                 size_factor=1.0):
        super().__init__()
        self.t_in  = t_in
        self.t_out = t_out
        self.dim   = dim

        self.gabor = GaborLayer(t_in, t_out, weight_scale, alpha, beta, freq_multiplier)
        self.mlp   = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape
        x_t = rearrange(x, 'b t c h w -> b c h w t')

        gabor_out = self.gabor(x_t)   # (B, C, H, W, T_out)
        mlp_out   = self.mlp(x_t)     # (B, C, H, W, T_out)

        # Fuse
        fused = torch.cat([gabor_out, mlp_out], dim=1)  # (B, 2C, H, W, T_out)
        fused = fused.permute(0, 1, 4, 2, 3)            # (B, 2C, T_out, H, W)
        fused = self.fusion(fused)                        # (B, C, T_out, H, W)
        fused = rearrange(fused, 'b c t h w -> b t c h w')

        # Gabor residual
        gabor_res = rearrange(gabor_out, 'b c h w t -> b t c h w')

        return fused + gabor_res


class WaveletGaborBlock(nn.Module):
    """Step 2 wrapper."""
    def __init__(self, t_in, t_out, dim,
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor=1.0,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in  = t_in
        self.t_out = t_out
        # Use _low params for the single full-res stream
        self.temporal = TemporalBlock(
            t_in, t_out, dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            size_factor,
        )

    def forward(self, x):
        return self.temporal(x)


class WaveletLASTOCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor=1.0,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out

        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.operator = WaveletGaborBlock(
            T_in, T_out, hidden_dim,
            num_blocks, sparsity_threshold, hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,
        )
        self.projection = nn.Sequential(
            TransformBlock(hidden_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1),
        )

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.lifting(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_in)
        x = self.operator(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.projection(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_out)
        return x


class WaveletLASTOCastForecaster(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor, total_steps, const_ratio,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.lastocast = WaveletLASTOCast(
            T_in, T_out, in_dim, hidden_dim,
            num_blocks, sparsity_threshold, hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,
        )
        self.T_in   = T_in
        self.T_out  = T_out
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        self.itr    = 0

    def forward(self, x, y=None, cmp_fft_loss=False):
        self.itr += 1
        return self.lastocast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            falfcl_loss = self.falfcl(xas, frames_gt)
            return xas, {'total_loss': falfcl_loss}
        return xas, None


def get_model(
    num_blocks=1, sparsity_threshold=0.01, hidden_size_factor=1,
    weight_scale_low=1.0, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.0, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    k_spatial=3, size_factor=1.0,
    total_steps=50000, const_ratio=0.1,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128), **kwargs
):
    return WaveletLASTOCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        num_blocks=num_blocks, sparsity_threshold=sparsity_threshold,
        hidden_size_factor=hidden_size_factor,
        weight_scale_low=weight_scale_low, alpha_low=alpha_low,
        beta_low=beta_low, freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high, freq_multiplier_high=freq_multiplier_high,
        k_spatial=k_spatial, size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )
