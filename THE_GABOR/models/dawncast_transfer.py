"""
DAWN-Cast with PER-SUBBAND Gabor parameters (Experiment 2 support).

Why this file exists
--------------------
The original `WGTMBlock` *derives* the HF Gabor frequency multipliers by
interpolating between `freq_multiplier_low` and `freq_multiplier_high` when
`hf_mode='separate'`:

    freq_mid = (freq_multiplier_low + freq_multiplier_high) / 2
    alpha_interp = i / (level - 1)
    freq_i = freq_multiplier_high * (1 - alpha_interp) + freq_mid * alpha_interp

so an individual subband (e.g. HF_level_2) cannot be addressed or initialised
independently.  Here every Gabor hyper-parameter is instead supplied as a
*sequence, one entry per subband*, ordered

    index 0 -> 'LL'
    index i -> 'HF_level_i'          (hf_mode='separate')
    index 1 -> 'HF_shared'           (hf_mode='shared')

A scalar is broadcast to every subband, so the previous behaviour is still
expressible; nothing is interpolated implicitly any more.

The ARCHITECTURE IS UNCHANGED.  Every component that is not part of this
parameter plumbing -- GaborLayer, FATBlock, SRST/STR, TransformBlock -- is
imported directly from the original `models/DAWNCast/dawncast.py`, which is
never modified, so the two models are literally the same code.
"""

from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse

# Imported unchanged from the original, untouched DAWN-Cast implementation.
from models.DAWNCast.dawncast import (FATBlock, GaborLayer, SRSTResBlock,
                                      STRModule, TransformBlock)
from utils.utilspp import RandomScheduling


# ============================================================
# Per-subband parameter plumbing
# ============================================================

def subband_names(level, hf_mode):
    """Canonical, stable subband order. Matches THE_GABOR's naming exactly."""
    assert hf_mode in ('shared', 'separate')
    if hf_mode == 'shared':
        return ['LL', 'HF_shared']
    return ['LL'] + [f'HF_level_{i + 1}' for i in range(level)]


def as_per_subband(value, names, param_name):
    """
    Broadcast a scalar to every subband, or validate a per-subband sequence.
    Also accepts a dict keyed by subband name.
    """
    if isinstance(value, dict):
        missing = [n for n in names if n not in value]
        if missing:
            raise ValueError(f"{param_name}: missing subbands {missing}")
        return [float(value[n]) for n in names]
    if isinstance(value, (list, tuple)):
        if len(value) != len(names):
            raise ValueError(
                f"{param_name}: got {len(value)} values for {len(names)} "
                f"subbands {names}")
        return [float(v) for v in value]
    return [float(value)] * len(names)


# ============================================================
# WGTM Block -- identical to the original except that Gabor
# hyper-parameters arrive per subband instead of interpolated.
# ============================================================

class WGTMBlockPerSubband(nn.Module):
    def __init__(self, t_in, t_out, dim, num_blocks, sparsity_threshold,
                 hidden_size_factor,
                 weight_scale, alpha, beta, freq_multiplier,
                 k_spatial, size_factor=1.0,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim = dim
        self.level = level
        self.hf_mode = hf_mode

        assert level in [1, 2, 3, 4], "Levels 1-4 supported"
        assert hf_mode in ['shared', 'separate']

        self.subbands = subband_names(level, hf_mode)
        ws = as_per_subband(weight_scale, self.subbands, 'weight_scale')
        al = as_per_subband(alpha, self.subbands, 'alpha')
        be = as_per_subband(beta, self.subbands, 'beta')
        fm = as_per_subband(freq_multiplier, self.subbands, 'freq_multiplier')
        self.gabor_config = OrderedDict(
            (n, dict(weight_scale=ws[i], alpha=al[i], beta=be[i],
                     freq_multiplier=fm[i]))
            for i, n in enumerate(self.subbands))

        self.wave = wave
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # ---- LL subband ----
        self.fat_ll = FATBlock(t_in, t_out, dim,
                               ws[0], al[0], be[0], fm[0], size_factor)

        # ---- HF subbands: one slot per subband, nothing interpolated ----
        if hf_mode == 'shared':
            self.fat_hf = FATBlock(t_in, t_out, 3 * dim,
                                   ws[1], al[1], be[1], fm[1], size_factor)
        else:
            self.fat_hf_streams = nn.ModuleList([
                FATBlock(t_in, t_out, 3 * dim,
                         ws[i + 1], al[i + 1], be[i + 1], fm[i + 1], size_factor)
                for i in range(level)
            ])

        # ---- SRST stack (unchanged) ----
        self.srst = nn.Sequential(
            SRSTResBlock(dim * t_out, num_blocks, sparsity_threshold,
                         hidden_size_factor, k_spatial),
            SRSTResBlock(dim * t_out, num_blocks, sparsity_threshold,
                         hidden_size_factor, k_spatial),
            STRModule(dim * t_out, num_blocks, sparsity_threshold,
                      hidden_size_factor=hidden_size_factor)
        )
        self.viz_counter = 0

    # ---- FAT block / Gabor accessors, keyed by canonical subband name ----
    def fat_blocks(self):
        out = OrderedDict()
        out['LL'] = self.fat_ll
        if self.hf_mode == 'shared':
            out['HF_shared'] = self.fat_hf
        else:
            for i, blk in enumerate(self.fat_hf_streams):
                out[f'HF_level_{i + 1}'] = blk
        return out

    def gabor_layers(self):
        return OrderedDict((n, b.gabor) for n, b in self.fat_blocks().items())

    # ---- forward: byte-for-byte the original WGTMBlock.forward ----
    def forward(self, x):
        B, T, C, H, W = x.shape

        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)

        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_gabor, ll_mlp, ll_fused = self.fat_ll(ll_t)

        hf_gabor_list, hf_fused_list, hf_mlp_list = [], [], []
        for i in range(len(hf_list)):
            hf_t = rearrange(hf_list[i], '(b t) c n h w -> b (c n) h w t', t=T)
            if self.hf_mode == 'shared':
                hf_gabor, hf_mlp, hf_fused = self.fat_hf(hf_t)
            else:
                hf_gabor, hf_mlp, hf_fused = self.fat_hf_streams[i](hf_t)
            hf_gabor_list.append(hf_gabor)
            hf_mlp_list.append(hf_mlp)
            hf_fused_list.append(hf_fused)

        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = [rearrange(h, 'b (c n) t h w -> (b t) c n h w', n=3)
                         for h in hf_fused_list]
        reconstructed = self.idwt((ll_recon, hf_recon_list))

        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_gabor_flat_list = [rearrange(h, 'b (c n) h w t -> (b t) c n h w', n=3)
                              for h in hf_gabor_list]
        gabor_residual = self.idwt((ll_gabor_flat, hf_gabor_flat_list))

        reconstructed = reconstructed[..., :H, :W]
        gabor_residual = gabor_residual[..., :H, :W]

        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)
        gabor_residual = rearrange(gabor_residual, '(b t) c h w -> b t c h w', t=self.t_out)

        x_srst = rearrange(reconstructed, 'b t c h w -> b h w (t c)')
        x_srst = self.srst(x_srst)
        x_srst = rearrange(x_srst, 'b h w (t c) -> b t c h w', t=self.t_out)

        return x_srst + gabor_residual


# ============================================================
# Full model / forecaster (identical structure to the original)
# ============================================================

class DAWNCastPerSubband(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim, num_blocks,
                 sparsity_threshold, hidden_size_factor,
                 weight_scale, alpha, beta, freq_multiplier,
                 k_spatial, size_factor=1.0, wave='haar', level=1,
                 hf_mode='shared'):
        super().__init__()
        self.T_in, self.T_out = T_in, T_out

        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.wgtm = WGTMBlockPerSubband(
            T_in, T_out, hidden_dim, num_blocks, sparsity_threshold,
            hidden_size_factor, weight_scale, alpha, beta, freq_multiplier,
            k_spatial, size_factor, wave, level, hf_mode,
        )
        self.projection = nn.Sequential(
            TransformBlock(hidden_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1),
        )

    def gabor_layers(self):
        return self.wgtm.gabor_layers()

    def fat_blocks(self):
        return self.wgtm.fat_blocks()

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.lifting(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_in)
        x = self.wgtm(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.projection(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_out)
        return x


class DAWNCastPerSubbandForecaster(nn.Module):
    """FACL-only forecaster, matching DAWNCastForecaster."""

    def __init__(self, T_in, T_out, in_dim, hidden_dim, num_blocks,
                 sparsity_threshold, hidden_size_factor,
                 weight_scale, alpha, beta, freq_multiplier,
                 size_factor, total_steps, const_ratio, k_spatial,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.dawncast = DAWNCastPerSubband(
            T_in, T_out, in_dim, hidden_dim, num_blocks, sparsity_threshold,
            hidden_size_factor, weight_scale, alpha, beta, freq_multiplier,
            k_spatial, size_factor, wave, level, hf_mode,
        )
        self.T_in, self.T_out = T_in, T_out
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        self.itr = 0

    def gabor_layers(self):
        return self.dawncast.gabor_layers()

    def fat_blocks(self):
        return self.dawncast.fat_blocks()

    def forward(self, x, y=None, cmp_fft_loss=False):
        self.itr += 1
        return self.dawncast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        pred = self(frames_in, frames_gt, compute_loss)
        if not compute_loss:
            return pred, None
        facl = self.falfcl(pred, frames_gt)
        return pred, {'facl_loss': facl, 'total_loss': facl}


def get_model(T_in=5, T_out=20, img_channels=4, dim=64,
              afno_blocks=4, sparsity_threshold=0.01, afno_hidden_size_factor=4,
              weight_scale=1.0, alpha=1.0, beta=1.0, freq_multiplier=4.0,
              size_factor=1.0, total_steps=50000, const_ratio=0.1, k_spatial=3,
              wave='db6', wavelet_level=2, hf_mode='separate', **kwargs):
    """
    `weight_scale` / `alpha` / `beta` / `freq_multiplier` accept a scalar
    (broadcast), a sequence ordered as `subband_names(wavelet_level, hf_mode)`,
    or a dict keyed by subband name.
    """
    return DAWNCastPerSubbandForecaster(
        T_in=T_in, T_out=T_out, in_dim=img_channels, hidden_dim=dim,
        num_blocks=afno_blocks, sparsity_threshold=sparsity_threshold,
        hidden_size_factor=afno_hidden_size_factor,
        weight_scale=weight_scale, alpha=alpha, beta=beta,
        freq_multiplier=freq_multiplier,
        size_factor=size_factor, total_steps=total_steps,
        const_ratio=const_ratio, k_spatial=k_spatial,
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )
