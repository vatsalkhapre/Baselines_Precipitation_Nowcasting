"""
Wavelet-Gabor LASTOCast Block — GFN variant

Replaces AFNO2D in conv_spectral with GlobalFilter (GFN).

Key differences vs AFNO:
    - GFN: pure learned complex multiplication in freq domain (no nonlinearity inside)
      → norm + activation AFTER irfft is critical (GFN is linear, stacking = 1 big linear op otherwise)
    - AFNO: 2-layer MLP in freq domain with ReLU + softshrink (nonlinear + sparse)
      → norm + activation still help but AFNO already has internal nonlinearity

GFN parameter count: h * w * dim * 2  (raw weights, no block structure)
    For dim=64*T_out, h=H, w=W//2+1: grows with spatial size.
    Start with num_gfn_layers=1, then increase.

conv_spectral structure (configurable):
    num_gfn_layers=1: [ResGFNBlock, ResGFNBlock, GlobalFilter]
    num_gfn_layers=2: [ResGFNBlock, ResGFNBlock, GlobalFilter, GlobalFilter]
    etc.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from utils.utilspp import RandomScheduling


# ============================================================
# Global Filter Network (GFN)
# Input/output: (B, H, W, C)  — same convention as AFNO
# ============================================================

class GlobalFilter(nn.Module):
    """
    Learned complex multiplication in 2D frequency domain.
    Input:  (B, H, W, C)
    Output: (B, H, W, C)  — with residual

    h, w must match the spatial size of the input.
    w should be H//2 + 1 (rfft2 output width).
    """
    def __init__(self, dim, h=8, w=5):
        super().__init__()
        # complex_weight: (h, w, dim, 2)  last dim = [real, imag]
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.h = h
        self.w = w

    def forward(self, x):
        # x: (B, H, W, C)
        bias = x
        B, H, W, C = x.shape
        x = x.float()

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')   # (B, H, W//2+1, C)

        weight = torch.view_as_complex(self.complex_weight)  # (h, w, C) complex
        x = x * weight                                        # broadcast over B

        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')  # (B, H, W, C)

        return x + bias


class SpectralBlock_GFN(nn.Module):
    """GFN + GroupNorm + SiLU. Norm/act after irfft is important since GFN is linear."""
    def __init__(self, dim, h, w, groups=8):
        super().__init__()
        self.proj = GlobalFilter(dim, h=h, w=w)
        self.norm = nn.GroupNorm(groups, dim)
        self.act  = nn.SiLU()

    def forward(self, x):
        # x: (B, H, W, C)
        x = self.proj(x)
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W) for GroupNorm
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)
        return x


class ResGFNBlock(nn.Module):
    """Two SpectralBlock_GFN layers with residual."""
    def __init__(self, dim, h, w, groups=8):
        super().__init__()
        self.block1 = SpectralBlock_GFN(dim, h, w, groups)
        self.block2 = SpectralBlock_GFN(dim, h, w, groups)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + x   # residual; dim unchanged so no projection needed


# ============================================================
# Helper: build conv_spectral with N GFN layers
# ============================================================

def build_gfn_spectral(dim, spatial_h, spatial_w, num_gfn_layers=1, groups=8):
    """
    Mirrors the AFNO conv_spectral structure:
        [ResGFNBlock, ResGFNBlock, GlobalFilter x num_gfn_layers]

    spatial_h, spatial_w: H and W of the feature map going into conv_spectral.
    GFN weight shape: (spatial_h, spatial_w//2+1, dim, 2)
    """
    gfn_w = spatial_w // 2 + 1
    layers = [
        ResGFNBlock(dim, spatial_h, gfn_w, groups),
        ResGFNBlock(dim, spatial_h, gfn_w, groups),
    ]
    for _ in range(num_gfn_layers):
        layers.append(GlobalFilter(dim, h=spatial_h, w=gfn_w))
    return nn.Sequential(*layers)


# ============================================================
# Unchanged: GaborLayer, Block, TransformBlock, BandTemporalStream
# ============================================================

class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0, freq_multiplier=1.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        self.freq = nn.Parameter(torch.rand(out_features))
        self.freq_multiplier = freq_multiplier

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.freq_multiplier * self.freq * self.linear(x)) * \
               torch.exp(-0.5 * D * self.gamma[None, :])


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


class BandTemporalStream(nn.Module):
    """Gabor + MLP dual-stream temporal modeling for one frequency band."""
    def __init__(self, t_in, t_out, dim, weight_scale, alpha, beta, freq_multiplier, size_factor=1.0):
        super().__init__()
        self.gabor  = GaborLayer(t_in, t_out, weight_scale, alpha, beta, freq_multiplier)
        self.mlp    = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W, T_in)
        gabor_out = self.gabor(x)                           # (B, C, H, W, T_out)
        mlp_out   = self.mlp(x)                             # (B, C, H, W, T_out)
        fused     = torch.cat([gabor_out, mlp_out], dim=1) # (B, 2C, H, W, T_out)
        fused     = fused.permute(0, 1, 4, 2, 3)           # (B, 2C, T_out, H, W)
        fused     = self.fusion(fused)                       # (B, C, T_out, H, W)
        return gabor_out, mlp_out, fused


# ============================================================
# Main Wavelet-GFN Block
# ============================================================

class WaveletGFNBlock(nn.Module):
    """
    Same as WaveletGaborBlock but conv_spectral uses GFN instead of AFNO.

    num_gfn_layers: number of raw GlobalFilter layers appended after the two ResGFNBlocks.
        1 → lightest (recommended to start)
        2, 3 → more capacity but quadratic param growth with spatial size
    input_shape: (H, W) of the input to the model (before lifting/DWT).
                 Used to compute GFN weight dimensions.
    """
    def __init__(self, t_in, t_out, dim,
                 num_gfn_layers,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared',
                 input_shape=(32, 32)):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim   = dim
        self.level = level
        self.hf_mode = hf_mode

        assert level in [1, 2, 3, 4]
        assert hf_mode in ['shared', 'separate']

        self.dwt  = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # LL stream
        self.stream_ll = BandTemporalStream(
            t_in, t_out, dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low, size_factor,
        )

        # HF streams
        if hf_mode == 'shared':
            self.stream_hf = BandTemporalStream(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high, size_factor,
            )
        else:
            self.hf_streams = nn.ModuleList()
            for i in range(level):
                freq_i = freq_multiplier_high if level == 1 else (
                    freq_multiplier_high * (1 - i / (level - 1))
                    + ((freq_multiplier_low + freq_multiplier_high) / 2) * (i / (level - 1))
                )
                self.hf_streams.append(BandTemporalStream(
                    t_in, t_out, 3 * dim,
                    weight_scale_high, alpha_high, beta_high, freq_i, size_factor,
                ))

        # GFN conv_spectral
        # The feature going in has shape (B, H, W, dim * t_out)
        spatial_h, spatial_w = input_shape
        self.conv_spectral = build_gfn_spectral(
            dim=dim * t_out,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            num_gfn_layers=num_gfn_layers,
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # 1. DWT
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)

        # 2. Temporal processing
        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_gabor, ll_mlp, ll_fused = self.stream_ll(ll_t)

        hf_gabor_list = []
        hf_fused_list = []
        hf_mlp_list   = []
        for i, hf in enumerate(hf_list):
            hf_t = rearrange(hf, '(b t) c n h w -> b (c n) h w t', t=T)
            if self.hf_mode == 'shared':
                hf_gabor, hf_mlp, hf_fused = self.stream_hf(hf_t)
            else:
                hf_gabor, hf_mlp, hf_fused = self.hf_streams[i](hf_t)
            hf_gabor_list.append(hf_gabor)
            hf_mlp_list.append(hf_mlp)
            hf_fused_list.append(hf_fused)

        # 3. IDWT — fused path
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = [
            rearrange(hf, 'b (c n) t h w -> (b t) c n h w', n=3)
            for hf in hf_fused_list
        ]
        reconstructed = self.idwt((ll_recon, hf_recon_list))

        # 4. IDWT — Gabor residual path
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_gabor_flat_list = [
            rearrange(hf, 'b (c n) h w t -> (b t) c n h w', n=3)
            for hf in hf_gabor_list
        ]
        gabor_residual = self.idwt((ll_gabor_flat, hf_gabor_flat_list))

        # 5. Trim
        reconstructed  = reconstructed[..., :H, :W]
        gabor_residual = gabor_residual[..., :H, :W]

        reconstructed  = rearrange(reconstructed,  '(b t) c h w -> b t c h w', t=self.t_out)
        gabor_residual = rearrange(gabor_residual, '(b t) c h w -> b t c h w', t=self.t_out)

        # 6. GFN Spatio-Temporal Interaction
        # conv_spectral expects (B, H, W, T*C)
        x_st = rearrange(reconstructed, 'b t c h w -> b h w (t c)')
        x_st = self.conv_spectral(x_st)
        x_st = rearrange(x_st, 'b h w (t c) -> b t c h w', t=self.t_out)

        return x_st + gabor_residual


# ============================================================
# Full model
# ============================================================

class WaveletLASTOCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 num_gfn_layers,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared',
                 input_shape=(32, 32)):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out

        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.operator = WaveletGFNBlock(
            T_in, T_out, hidden_dim,
            num_gfn_layers,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            size_factor, wave, level, hf_mode,
            input_shape=input_shape,
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
                 num_gfn_layers,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor, total_steps, const_ratio,
                 wave='haar', level=1, hf_mode='shared',
                 input_shape=(32, 32)):
        super().__init__()
        self.lastocast = WaveletLASTOCast(
            T_in, T_out, in_dim, hidden_dim,
            num_gfn_layers,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            size_factor, wave, level, hf_mode, input_shape,
        )
        self.T_in   = T_in
        self.T_out  = T_out
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        

    def forward(self, x, y=None, cmp_fft_loss=False):
        
        return self.lastocast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            falfcl_loss = self.falfcl(xas, frames_gt)
            return xas, {'total_loss': falfcl_loss}
        return xas, None


# ============================================================
# Model Factory
# ============================================================

def get_model(
    num_gfn_layers=1,
    weight_scale_low=1.5, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.5, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    size_factor=1.0,
    total_steps=50000, const_ratio=0.5,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(32, 32),
    **kwargs
):
    return WaveletLASTOCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        num_gfn_layers=num_gfn_layers,
        weight_scale_low=weight_scale_low, alpha_low=alpha_low,
        beta_low=beta_low, freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high, freq_multiplier_high=freq_multiplier_high,
        size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
        input_shape=input_shape,
    )


# ============================================================
# Self-test  (also prints param counts per num_gfn_layers)
# ============================================================
if __name__ == '__main__':
    print("=== Wavelet-GFN LASTOCast Self-Test ===\n")

    configs = [
        {'wave': 'db6', 'level': l, 'hf_mode': m, 'num_gfn_layers': n}
        for l in [2, 3]
        for m in ['shared', 'separate']
        for n in [1, 2, 3]
    ]

    passed = failed = 0
    for cfg in configs:
        tag = f"J{cfg['level']}_{cfg['hf_mode']}_gfn{cfg['num_gfn_layers']}"
        try:
            model = get_model(
                img_channels=4, dim=64,
                T_in=5, T_out=20,
                wave=cfg['wave'], wavelet_level=cfg['level'],
                hf_mode=cfg['hf_mode'],
                num_gfn_layers=cfg['num_gfn_layers'],
                input_shape=(32, 32),
            )
            x   = torch.randn(2, 5, 4, 32, 32)
            out = model(x)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            assert out.shape == (2, 20, 4, 32, 32), f"Shape mismatch: {out.shape}"
            print(f"  [PASS] {tag:<30} | out={tuple(out.shape)} | {params:.2f}M")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {tag:<30} | {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(configs)} configs")
