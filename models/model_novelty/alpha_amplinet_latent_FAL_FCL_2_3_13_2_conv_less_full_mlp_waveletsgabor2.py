"""
Wavelet-Gabor LASTOCast Block (Unified)

Supports:
    - J=1: 2 Gabors (LL + HF)
    - J=2, mode='shared': 2 Gabors (LL + all HF pooled)  [Option A]
    - J=2, mode='separate': 3 Gabors (LL + HF-L1 + HF-L2) [Option B]

Requirements:
    pip install pytorch_wavelets
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from utils.utilspp import RandomScheduling


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
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class TransformBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups,
                            kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups=groups,
                            kernel_size=kernel_size, padding_mode=padding_mode)
        self.skip = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


# ============================================================
# Temporal stream: Gabor + MLP + Fusion for a single band
# ============================================================

class BandTemporalStream(nn.Module):
    """Applies Gabor+MLP dual-stream temporal modeling to a single frequency band."""
    def __init__(self, t_in, t_out, dim, weight_scale, alpha, beta,
                 freq_multiplier, size_factor=1.0):
        super().__init__()
        self.gabor = GaborLayer(t_in, t_out, weight_scale, alpha, beta, freq_multiplier)
        self.mlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, H, W, T_in)
        returns: gabor_out (B, C, H, W, T_out), fused_out (B, C, T_out, H, W)
        """
        gabor_out = self.gabor(x)   # (B, C, H, W, T_out)
        mlp_out = self.mlp(x)       # (B, C, H, W, T_out)

        # Fuse: cat along channel, permute for Conv3d
        fused = torch.cat([gabor_out, mlp_out], dim=1)  # (B, 2C, H, W, T_out)
        fused = fused.permute(0, 1, 4, 2, 3)            # (B, 2C, T_out, H, W)
        fused = self.fusion(fused)                        # (B, C, T_out, H, W)

        return gabor_out, fused


# ============================================================
# Main Wavelet-Gabor Block
# ============================================================

class WaveletGaborBlock(nn.Module):
    """
    LASTOCast block with wavelet-decomposed dual Gabor temporal modeling.

    Args:
        level: DWT decomposition level (1 or 2)
        hf_mode: 'shared' or 'separate' (only matters when level=2)
            - 'shared': single Gabor for all HF bands (Option A)
            - 'separate': one Gabor per HF level (Option B)
    """
    def __init__(self, t_in, t_out, dim,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim = dim
        self.level = level
        self.hf_mode = hf_mode

        assert level in [1, 2], "Only level 1 and 2 supported"
        assert hf_mode in ['shared', 'separate'], "hf_mode must be 'shared' or 'separate'"

        # ---- Wavelet transform ----
        self.wave = wave
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # ---- LL temporal stream (always present) ----
        self.stream_ll = BandTemporalStream(
            t_in, t_out, dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            size_factor,
        )

        # ---- HF temporal streams ----
        if level == 1 or hf_mode == 'shared':
            # Single stream for all HF bands (3*dim channels)
            self.stream_hf = BandTemporalStream(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                size_factor,
            )
        else:
            # level=2, separate: one stream per HF level
            # Level 1 HF: coarser details (at LL resolution / 2)
            self.stream_hf_l1 = BandTemporalStream(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                size_factor,
            )
            # Level 2 HF: finer details (at input resolution / 2)
            # Use slightly different freq — midpoint between low and high
            freq_mid = (freq_multiplier_low + freq_multiplier_high) / 2
            self.stream_hf_l2 = BandTemporalStream(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_mid,
                size_factor,
            )

        # ---- Spatio-Temporal Interaction ----
        self.spatial_temporal = nn.Sequential(
            TransformBlock(dim * t_out, dim * t_out),
            TransformBlock(dim * t_out, dim * t_out),
            nn.Conv2d(dim * t_out, dim * t_out, kernel_size=3, padding=1),
        )

    def _process_j1(self, x):
        """Process with J=1 decomposition."""
        B, T, C, H, W = x.shape

        # DWT
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)
        hf = hf_list[0]  # (B*T, C, 3, H', W')

        # Reshape for temporal processing
        ll = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        hf = rearrange(hf, '(b t) c n h w -> b (c n) h w t', t=T)

        # Temporal streams
        ll_gabor, ll_fused = self.stream_ll(ll)
        hf_gabor, hf_fused = self.stream_hf(hf)

        # IDWT reconstruction (fused path)
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon = rearrange(hf_fused, 'b (c n) t h w -> (b t) c n h w', n=3)
        reconstructed = self.idwt((ll_recon, [hf_recon]))

        # IDWT reconstruction (gabor-only residual)
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_gabor_flat = rearrange(hf_gabor, 'b (c n) h w t -> (b t) c n h w', n=3)
        gabor_residual = self.idwt((ll_gabor_flat, [hf_gabor_flat]))

        return reconstructed, gabor_residual, H, W

    def _process_j2_shared(self, x):
        """Process with J=2, shared HF Gabor (Option A).
        
        Uses the SAME Gabor+MLP stream for both HF levels.
        Each HF level is processed independently, then reconstructed via proper J=2 IDWT.
        """
        B, T, C, H, W = x.shape

        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)
        # ll: (B*T, C, H/4, W/4)
        # hf_list[0]: (B*T, C, 3, H/4, W/4)  — coarse HF
        # hf_list[1]: (B*T, C, 3, H/2, W/2)  — fine HF

        hf_l1 = hf_list[0]  # coarse
        hf_l2 = hf_list[1]  # fine

        # Reshape for temporal processing
        ll = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        hf_l1_t = rearrange(hf_l1, '(b t) c n h w -> b (c n) h w t', t=T)
        hf_l2_t = rearrange(hf_l2, '(b t) c n h w -> b (c n) h w t', t=T)

        # LL gets its own stream
        ll_gabor, ll_fused = self.stream_ll(ll)

        # Both HF levels share the SAME stream
        hf_l1_gabor, hf_l1_fused = self.stream_hf(hf_l1_t)
        hf_l2_gabor, hf_l2_fused = self.stream_hf(hf_l2_t)

        # Proper J=2 IDWT reconstruction (fused path)
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_l1_recon = rearrange(hf_l1_fused, 'b (c n) t h w -> (b t) c n h w', n=3)
        hf_l2_recon = rearrange(hf_l2_fused, 'b (c n) t h w -> (b t) c n h w', n=3)
        reconstructed = self.idwt((ll_recon, [hf_l1_recon, hf_l2_recon]))

        # Gabor residual (gabor-only path)
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_l1_gabor_flat = rearrange(hf_l1_gabor, 'b (c n) h w t -> (b t) c n h w', n=3)
        hf_l2_gabor_flat = rearrange(hf_l2_gabor, 'b (c n) h w t -> (b t) c n h w', n=3)
        gabor_residual = self.idwt((ll_gabor_flat, [hf_l1_gabor_flat, hf_l2_gabor_flat]))

        return reconstructed, gabor_residual, H, W

    def _process_j2_separate(self, x):
        """Process with J=2, separate HF Gabors (Option B)."""
        B, T, C, H, W = x.shape

        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)
        # ll: (B*T, C, H/4, W/4)
        # hf_list[0]: (B*T, C, 3, H/4, W/4)  — level 1 (coarse)
        # hf_list[1]: (B*T, C, 3, H/2, W/2)  — level 2 (fine)

        hf_l1 = hf_list[0]
        hf_l2 = hf_list[1]

        # Reshape for temporal
        ll = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        hf_l1_t = rearrange(hf_l1, '(b t) c n h w -> b (c n) h w t', t=T)
        hf_l2_t = rearrange(hf_l2, '(b t) c n h w -> b (c n) h w t', t=T)

        # Three separate temporal streams
        ll_gabor, ll_fused = self.stream_ll(ll)
        hf_l1_gabor, hf_l1_fused = self.stream_hf_l1(hf_l1_t)
        hf_l2_gabor, hf_l2_fused = self.stream_hf_l2(hf_l2_t)

        # Proper J=2 IDWT reconstruction
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_l1_recon = rearrange(hf_l1_fused, 'b (c n) t h w -> (b t) c n h w', n=3)
        hf_l2_recon = rearrange(hf_l2_fused, 'b (c n) t h w -> (b t) c n h w', n=3)
        reconstructed = self.idwt((ll_recon, [hf_l1_recon, hf_l2_recon]))

        # Gabor residual
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_l1_gabor_flat = rearrange(hf_l1_gabor, 'b (c n) h w t -> (b t) c n h w', n=3)
        hf_l2_gabor_flat = rearrange(hf_l2_gabor, 'b (c n) h w t -> (b t) c n h w', n=3)
        gabor_residual = self.idwt((ll_gabor_flat, [hf_l1_gabor_flat, hf_l2_gabor_flat]))

        return reconstructed, gabor_residual, H, W

    def forward(self, x):
        # x: (B, T_in, C, H, W)

        # ---- Wavelet + Temporal processing ----
        if self.level == 1:
            reconstructed, gabor_residual, H, W = self._process_j1(x)
        elif self.hf_mode == 'shared':
            reconstructed, gabor_residual, H, W = self._process_j2_shared(x)
        else:
            reconstructed, gabor_residual, H, W = self._process_j2_separate(x)

        # Trim to original spatial size
        reconstructed = reconstructed[..., :H, :W]
        gabor_residual = gabor_residual[..., :H, :W]

        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)
        gabor_residual = rearrange(gabor_residual, '(b t) c h w -> b t c h w', t=self.t_out)

        # ---- Spatio-Temporal Interaction ----
        x_st = rearrange(reconstructed, 'b t c h w -> b (t c) h w')
        x_st = self.spatial_temporal(x_st)
        x_st = rearrange(x_st, 'b (t c) h w -> b t c h w', t=self.t_out)

        # ---- Residual from Gabor path ----
        x = x_st + gabor_residual

        return x


# ============================================================
# Full LASTOCast with Wavelet-Gabor
# ============================================================

class WaveletLASTOCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out

        # Lifting
        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # Core operator
        self.operator = WaveletGaborBlock(
            T_in, T_out, hidden_dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            size_factor, wave, level, hf_mode,
        )

        # Projection
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
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor, total_steps, const_ratio,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.lastocast = WaveletLASTOCast(
            T_in, T_out, in_dim, hidden_dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            size_factor, wave, level, hf_mode,
        )
        self.T_in = T_in
        self.T_out = T_out
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        self.itr = 0

    def forward(self, x, y=None, cmp_fft_loss=False):
        self.itr += 1
        return self.lastocast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            falfcl_loss = self.falfcl(xas, frames_gt)
            loss = {'total_loss': falfcl_loss}
            return xas, loss
        else:
            return xas, None


# ============================================================
# Model Factory
# ============================================================

def get_model(
    weight_scale_low=1.5, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.5, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    size_factor=1.0,
    total_steps=50000, const_ratio=0.5,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128),
    **kwargs
):
    model = WaveletLASTOCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        weight_scale_low=weight_scale_low, alpha_low=alpha_low,
        beta_low=beta_low, freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high, freq_multiplier_high=freq_multiplier_high,
        size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )
    return model


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':

    configs = [
        {'level': 1, 'hf_mode': 'shared',   'label': 'J=1'},
        {'level': 2, 'hf_mode': 'shared',   'label': 'J=2 Option A (shared)'},
        {'level': 2, 'hf_mode': 'separate', 'label': 'J=2 Option B (separate)'},
    ]

    for wave in ['haar', 'db2', 'db3', 'coif1']:
        for cfg in configs:
            try:
                model = get_model(
                    img_channels=4, dim=64,
                    T_in=5, T_out=20,
                    wave=wave, level=cfg['level'], hf_mode=cfg['hf_mode'],
                )
                x = torch.randn(2, 5, 4, 32, 32)
                out = model(x)
                params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
                assert out.shape == (2, 20, 4, 32, 32)
                print(f"  [PASS] {wave} | {cfg['label']:<25} | out={tuple(out.shape)} | {params:.2f}M")
            except Exception as e:
                print(f"  [FAIL] {wave} | {cfg['label']:<25} | {e}")

    print("\nAll tests complete!")