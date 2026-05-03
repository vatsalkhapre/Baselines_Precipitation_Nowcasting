"""
Wavelet-Gabor LASTOCast — No Wavelet, No Gabor
Single MLP for temporal modeling at full resolution.
conv_spectral (AFNO + dw conv + channel mixing), Lifting, Projection unchanged.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from utils.utilspp import RandomScheduling


# ============================================================
# AFNO2D — unchanged
# ============================================================

class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=1, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0
        self.hidden_size        = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks         = num_blocks
        self.block_size         = hidden_size // num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x
        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape
        N = H * W
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros_like(o2_real)

        total_modes = N // 2 + 1
        kept_modes  = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + self.b1[0]
        )
        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + self.b1[1]
        )
        o2_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + self.b2[0]
        )
        o2_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)
        return x + bias


# ============================================================
# SpectralBlock_2D, ResneSpectralBlock — unchanged
# ============================================================

class SpectralBlock_2D(nn.Module):
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 k_spatial, groupnorm=True, groups=8):
        super().__init__()
        pad_spatial = (k_spatial - 1) // 2
        self.proj        = AFNO2D(dim, num_blocks, sparsity_threshold,
                                  hidden_size_factor=hidden_size_factor)
        self.dw_spatial  = nn.Conv2d(dim, dim, kernel_size=k_spatial,
                                     padding=pad_spatial, groups=dim, bias=False)
        self.norm        = nn.GroupNorm(groups, dim) if groupnorm else nn.BatchNorm2d(dim)
        self.pw          = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        x_ = x.permute(0, 3, 1, 2)
        x_spa  = self.dw_spatial(x_)
        x_spec = self.proj(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_fused = x_spa + x_spec
        x_fused = self.norm(x_fused)
        x_fused = self.act(x_fused)
        x_fused = self.pw(x_fused)
        return x_fused.permute(0, 2, 3, 1)


class ResneSpectralBlock(nn.Module):
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 k_spatial, groups=8):
        super().__init__()
        self.block1   = SpectralBlock_2D(dim, num_blocks, sparsity_threshold,
                                         hidden_size_factor, k_spatial, groups=groups)
        self.block2   = SpectralBlock_2D(dim, num_blocks, sparsity_threshold,
                                         hidden_size_factor, k_spatial, groups=groups)
        self.res_conv = nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


# ============================================================
# Block, TransformBlock — unchanged
# ============================================================

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


# ============================================================
# WaveletGaborBlock — No Wavelet, No Gabor
# Single MLP at full resolution + conv_spectral
# ============================================================

class WaveletGaborBlock(nn.Module):
    """
    No Wavelet, No Gabor.
    Single MLP temporal stream at full resolution.
    conv_spectral (AFNO + dw conv + channel mixing) unchanged.
    No gabor residual — MLP output feeds directly into conv_spectral.
    """
    def __init__(self, t_in, t_out, dim,
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in  = t_in
        self.t_out = t_out
        self.dim   = dim

        # Single MLP: T_in → T_out, applied per spatial location
        self.mlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )

        # conv_spectral unchanged
        self.conv_spectral = nn.Sequential(
            ResneSpectralBlock(dim * t_out, num_blocks, sparsity_threshold, hidden_size_factor, k_spatial),
            ResneSpectralBlock(dim * t_out, num_blocks, sparsity_threshold, hidden_size_factor, k_spatial),
            AFNO2D(dim * t_out, num_blocks, sparsity_threshold, hidden_size_factor=hidden_size_factor),
        )

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape

        # MLP temporal modeling at full resolution
        x_t  = rearrange(x, 'b t c h w -> b c h w t')   # (B, C, H, W, T_in)
        x_t  = self.mlp(x_t)                              # (B, C, H, W, T_out)
        x_out = rearrange(x_t, 'b c h w t -> b t c h w') # (B, T_out, C, H, W)

        # conv_spectral spatio-temporal interaction
        x_st = rearrange(x_out, 'b t c h w -> b h w (t c)')
        x_st = self.conv_spectral(x_st)
        x_st = rearrange(x_st, 'b h w (t c) -> b t c h w', t=self.t_out)

        return x_st


# ============================================================
# Full model — unchanged
# ============================================================

class WaveletLASTOCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
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
                 size_factor, total_steps, const_ratio,
                 k_spatial, wave='haar', level=1, hf_mode='shared'):
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


# ============================================================
# Model Factory — same signature as original
# ============================================================

def get_model(
    afno_blocks, sparsity_threshold, afno_hidden_size_factor,
    weight_scale_low=1.5, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.5, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    size_factor=1.0, total_steps=50000, const_ratio=0.5, k_spatial=3,
    img_channels=1, dim=64, T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128), **kwargs
):
    return WaveletLASTOCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        num_blocks=afno_blocks, sparsity_threshold=sparsity_threshold,
        hidden_size_factor=afno_hidden_size_factor,
        weight_scale_low=weight_scale_low, alpha_low=alpha_low,
        beta_low=beta_low, freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high, freq_multiplier_high=freq_multiplier_high,
        size_factor=size_factor, total_steps=total_steps, const_ratio=const_ratio,
        k_spatial=k_spatial, wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )
