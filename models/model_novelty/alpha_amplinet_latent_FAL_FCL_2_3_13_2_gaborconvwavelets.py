"""
Wavelet LASTOCast V3

Pipeline:
    Input → Lifting → DWT → [Gabor+MLP → Fusion → S-T Conv] per band → IDWT → Residual → Projection

Each wavelet band gets the FULL operator pipeline independently.

Supports:
    - J=1: 2 independent pipelines (LL + HF)
    - J=2, shared: 2 pipelines (LL + shared HF for both levels)
    - J=2, separate: 3 pipelines (LL + HF-L1 + HF-L2)
    - residual_mode: 'gabor', 'mlp', 'none'

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


# ============================================================
# Building Blocks
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
# Band Pipeline: Gabor + MLP → Fusion → S-T Conv (complete per band)
# ============================================================

class BandPipeline(nn.Module):
    """Complete processing pipeline for a single wavelet band.
    
    Gabor + MLP (temporal) → Fusion → Spatio-Temporal Conv
    
    Args:
        t_in, t_out: temporal input/output lengths
        dim: channel dimension (C for LL, 3*C for HF)
        weight_scale, alpha, beta, freq_multiplier: Gabor params
        size_factor: MLP hidden size multiplier
    """
    def __init__(self, t_in, t_out, dim, weight_scale, alpha, beta,
                 freq_multiplier, size_factor=1.0):
        super().__init__()
        self.t_out = t_out

        # Spectral Temporal Modeling
        self.gabor = GaborLayer(t_in, t_out, weight_scale, alpha, beta, freq_multiplier)
        self.mlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )

        # Spectro-Temporal Fusion
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

        # Spatio-Temporal Interaction
        st_channels = dim * t_out
        self.spatial_temporal = nn.Sequential(
            TransformBlock(st_channels, st_channels),
            TransformBlock(st_channels, st_channels),
            nn.Conv2d(st_channels, st_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        x: (B, C, H, W, T_in)
        returns:
            gabor_out: (B, C, H, W, T_out) — for residual
            mlp_out: (B, C, H, W, T_out) — for residual
            processed: (B*T_out, C, H, W) — after full pipeline
        """
        # Temporal modeling
        gabor_out = self.gabor(x)     # (B, C, H, W, T_out)
        mlp_out = self.mlp(x)         # (B, C, H, W, T_out)

        # Fusion
        fused = torch.cat([gabor_out, mlp_out], dim=1)  # (B, 2C, H, W, T_out)
        fused = fused.permute(0, 1, 4, 2, 3)            # (B, 2C, T_out, H, W)
        fused = self.fusion(fused)                        # (B, C, T_out, H, W)
        fused = fused.permute(0, 2, 1, 3, 4)            # (B, T_out, C, H, W)

        # Spatio-Temporal Conv
        x_st = rearrange(fused, 'b t c h w -> b (t c) h w')
        x_st = self.spatial_temporal(x_st)
        processed = rearrange(x_st, 'b (t c) h w -> (b t) c h w', t=self.t_out)

        return gabor_out, mlp_out, processed


# ============================================================
# Core Operator Block
# ============================================================

class WaveletGaborBlockV3(nn.Module):
    """
    LASTOCast block with full pipeline per wavelet band.

    Pipeline:
        1. DWT decomposition
        2. Full [Gabor+MLP → Fusion → S-T Conv] per band
        3. IDWT reconstruction
        4. Residual connection

    Args:
        level: DWT decomposition level (1 or 2)
        hf_mode: 'shared' or 'separate' (for J=2)
        residual_mode: 'gabor', 'mlp', or 'none'
    """
    def __init__(self, t_in, t_out, dim,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor=1.0, wave='haar', level=1,
                 hf_mode='shared', residual_mode='gabor'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim = dim
        self.level = level
        self.wave = wave
        self.hf_mode = hf_mode
        self.residual_mode = residual_mode

        assert level in [1, 2]
        assert hf_mode in ['shared', 'separate']
        assert residual_mode in ['gabor', 'mlp', 'none']

        # ---- Wavelet transform ----
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # ---- LL pipeline (always present) ----
        self.pipeline_ll = BandPipeline(
            t_in, t_out, dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            size_factor,
        )

        # ---- HF pipeline(s) ----
        if level == 1 or hf_mode == 'shared':
            self.pipeline_hf = BandPipeline(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                size_factor,
            )
        else:
            # J=2 separate: independent pipeline per HF level
            self.pipeline_hf_l1 = BandPipeline(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                size_factor,
            )
            freq_mid = (freq_multiplier_low + freq_multiplier_high) / 2
            self.pipeline_hf_l2 = BandPipeline(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_mid,
                size_factor,
            )

    def _process_band(self, pipeline, band_t):
        """
        Run a band through its pipeline.
        band_t: (B, C, H, W, T_in)
        returns: gabor_out, mlp_out, processed (B*T_out, C, H, W)
        """
        return pipeline(band_t)

    def _build_residual(self, ll_component, hf_components):
        """Reconstruct residual from per-band Gabor/MLP outputs via IDWT."""
        # ll_component: (B, C, H', W', T_out) → (B*T_out, C, H', W')
        ll_flat = rearrange(ll_component, 'b c h w t -> (b t) c h w')

        hf_flat_list = []
        for hf_comp in hf_components:
            # hf_comp: (B, 3C, H', W', T_out) → (B*T_out, C, 3, H', W')
            hf_flat = rearrange(hf_comp, 'b (c n) h w t -> (b t) c n h w', n=3)
            hf_flat_list.append(hf_flat)

        residual = self.idwt((ll_flat, hf_flat_list))
        return residual

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape

        # ============================================================
        # 1. DWT decomposition
        # ============================================================
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)

        # Reshape for temporal processing (move T to last dim)
        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)

        # ============================================================
        # 2. Full pipeline per band
        # ============================================================

        # --- LL band ---
        ll_gabor, ll_mlp, ll_processed = self._process_band(self.pipeline_ll, ll_t)

        # --- HF band(s) ---
        hf_gabor_list = []
        hf_mlp_list = []
        hf_processed_list = []

        if self.level == 1:
            hf_t = rearrange(hf_list[0], '(b t) c n h w -> b (c n) h w t', t=T)
            hf_gabor, hf_mlp, hf_proc = self._process_band(self.pipeline_hf, hf_t)
            hf_gabor_list.append(hf_gabor)
            hf_mlp_list.append(hf_mlp)
            # Restore DWT format: (B*T_out, C, 3, H', W')
            hf_proc = rearrange(hf_proc, 'bt (c n) h w -> bt c n h w', n=3)
            hf_processed_list.append(hf_proc)

        elif self.hf_mode == 'shared':
            # Both HF levels through same pipeline
            for i in range(len(hf_list)):
                hf_t = rearrange(hf_list[i], '(b t) c n h w -> b (c n) h w t', t=T)
                hf_gabor, hf_mlp, hf_proc = self._process_band(self.pipeline_hf, hf_t)
                hf_gabor_list.append(hf_gabor)
                hf_mlp_list.append(hf_mlp)
                hf_proc = rearrange(hf_proc, 'bt (c n) h w -> bt c n h w', n=3)
                hf_processed_list.append(hf_proc)

        else:
            # J=2 separate: each HF level gets its own pipeline
            pipelines = [self.pipeline_hf_l1, self.pipeline_hf_l2]
            for i in range(len(hf_list)):
                hf_t = rearrange(hf_list[i], '(b t) c n h w -> b (c n) h w t', t=T)
                hf_gabor, hf_mlp, hf_proc = self._process_band(pipelines[i], hf_t)
                hf_gabor_list.append(hf_gabor)
                hf_mlp_list.append(hf_mlp)
                hf_proc = rearrange(hf_proc, 'bt (c n) h w -> bt c n h w', n=3)
                hf_processed_list.append(hf_proc)

        # ============================================================
        # 3. IDWT reconstruction
        # ============================================================
        reconstructed = self.idwt((ll_processed, hf_processed_list))
        reconstructed = reconstructed[..., :H, :W]
        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)

        # ============================================================
        # 4. Residual connection
        # ============================================================
        if self.residual_mode == 'none':
            return reconstructed

        # Build residual via IDWT from per-band Gabor or MLP outputs
        if self.residual_mode == 'gabor':
            residual = self._build_residual(ll_gabor, hf_gabor_list)
        else:  # mlp
            residual = self._build_residual(ll_mlp, hf_mlp_list)

        residual = residual[..., :H, :W]
        residual = rearrange(residual, '(b t) c h w -> b t c h w', t=self.t_out)

        return reconstructed + residual


# ============================================================
# Full Model
# ============================================================

class WaveletLASTOCastV3(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor=1.0, wave='haar', level=1,
                 hf_mode='shared', residual_mode='gabor'):
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
        self.operator = WaveletGaborBlockV3(
            T_in, T_out, hidden_dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            size_factor, wave, level, hf_mode, residual_mode,
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


class WaveletLASTOCastV3Forecaster(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor, total_steps, const_ratio,
                 wave='haar', level=1, hf_mode='shared', residual_mode='gabor'):
        super().__init__()
        self.lastocast = WaveletLASTOCastV3(
            T_in, T_out, in_dim, hidden_dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            size_factor, wave, level, hf_mode, residual_mode,
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
    wave='haar', wavelet_level=1, hf_mode='shared', residual_mode='gabor',
    input_shape=(128, 128),
    **kwargs
):
    model = WaveletLASTOCastV3Forecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        weight_scale_low=weight_scale_low, alpha_low=alpha_low,
        beta_low=beta_low, freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high, freq_multiplier_high=freq_multiplier_high,
        size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        wave=wave, level=wavelet_level, hf_mode=hf_mode, residual_mode=residual_mode,
    )
    return model


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    print("=== Wavelet LASTOCast V3 Self-Test ===\n")

    configs = []
    for wave in ['haar', 'db4', 'db6']:
        for level in [1, 2]:
            for hf_mode in ['shared', 'separate']:
                for res_mode in ['gabor', 'mlp', 'none']:
                    # skip separate for J=1 (only 1 HF level)
                    if level == 1 and hf_mode == 'separate':
                        continue
                    configs.append({
                        'wave': wave, 'level': level,
                        'hf_mode': hf_mode, 'residual_mode': res_mode,
                    })

    passed = 0
    failed = 0
    for cfg in configs:
        tag = f"{cfg['wave']}_J{cfg['level']}_{cfg['hf_mode']}_{cfg['residual_mode']}"
        try:
            model = get_model(
                img_channels=4, dim=64,
                T_in=5, T_out=20,
                wave=cfg['wave'], level=cfg['level'],
                hf_mode=cfg['hf_mode'], residual_mode=cfg['residual_mode'],
            )
            x = torch.randn(2, 5, 4, 32, 32)
            out = model(x)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            assert out.shape == (2, 20, 4, 32, 32), f"Shape mismatch: {out.shape}"
            print(f"  [PASS] {tag:<40} | out={tuple(out.shape)} | {params:.2f}M")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {tag:<40} | {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(configs)} configs")
