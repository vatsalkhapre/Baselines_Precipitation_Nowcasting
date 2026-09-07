"""
DAWN-Cast ablation variants (Part B / Part C).

One factory, `get_model(ablation=...)`, returns the full DAWN-Cast with exactly
one component removed.  Everything not being ablated is imported unchanged from
`models/DAWNCast/dawncast.py` (never modified) or reused from
`dawncast_transfer.py`, so each variant differs from the baseline in exactly one
place.

    ablation key      what is removed                                    new code?
    ----------------  ------------------------------------------------   ---------
    'none'            nothing (baseline, per-subband Gabor params)       no
    'a_no_wavelet'    DWT/IDWT; one FAT block on full-resolution feats   yes
    'b_shared_fat'    separate HF FAT blocks -> a single shared one      no (hf_mode='shared')
    'c_no_gabor'      Gabor stream -> Linear(T_in, T_out)                yes
    'd_no_str'        `str_branch` (STRModule) INSIDE each SRSTBlock     yes
    'e_no_spatial'    `spatial_branch` (Conv2d) INSIDE each SRSTBlock    yes
    'f_no_refinement' whole SRST stack -> use Stage-1 model instead      no (run_pixel.py)
    'f_no_srst'       whole `self.srst` Sequential, residual path kept   yes
    'g_no_wgtm'       whole WGTM block -> a single Linear(T_in, T_out)   yes

Notes
-----
* (b) and (f) need no ablation code: (b) is `hf_mode='shared'` on the baseline,
  and (f) is the Stage-1 model (`THE_GABOR/models/gabor_mlp_model.py`), whose
  numbers come from Stage-1 training.
* `f_no_srst` vs `f_no_refinement`: BOTH drop the SRST stack, but they are not the
  same model. `f_no_srst` removes ONLY `self.srst`, so the WGTM output becomes
  `reconstructed + gabor_residual` -- the Gabor residual bypass is retained, which
  makes it a strict "DAWN-Cast minus refinement" ablation. The Stage-1 model also
  omits that residual path, so `f_no_refinement` is "minus SRST AND minus the
  Gabor residual". Use `f_no_srst` when the clean single-component ablation is
  wanted; use `f_no_refinement` only to reuse Stage-1 numbers for free.
* (d)/(e) remove the branch INSIDE each SRSTBlock, leaving the other branch,
  the GroupNorm, the SiLU and the channel mixing in place.  The top-level
  `STRModule` that follows the two SRSTResBlocks is left untouched.
* (c) keeps the dual-stream structure so the fusion Conv3d still sees 2*dim
  channels; only the Gabor stream is swapped for a plain Linear(T_in, T_out).
"""

from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse

from models.DAWNCast.dawncast import (FATBlock, GaborLayer, STRModule,
                                      TransformBlock)
from THE_GABOR.models.dawncast_transfer import as_per_subband, subband_names
from utils.utilspp import RandomScheduling

ABLATIONS = ('none', 'a_no_wavelet', 'b_shared_fat', 'c_no_gabor',
             'd_no_str', 'e_no_spatial', 'f_no_refinement', 'f_no_srst',
             'g_no_wgtm')


def _fit_per_subband(value, names, param_name):
    """
    Size a per-subband hyper-parameter list to THIS ablation's subband list.

    Ablations change how many subbands exist -- `a_no_wavelet` has only ['LL'],
    `b_shared_fat` has ['LL', 'HF_shared'] -- while the caller supplies the
    baseline-length list built from *_low/*_high (e.g. 3 entries for db4 J=2).
    Truncating from the front keeps the intended meaning: index 0 is always LL,
    index 1 is the first/highest HF setting, which is what a shared HF block
    should inherit.  Scalars and dicts fall through to the strict helper.
    """
    if isinstance(value, (list, tuple)) and len(value) != len(names):
        trimmed = list(value)[:len(names)]
        while len(trimmed) < len(names):
            trimmed.append(trimmed[-1])
        print(f'[ablation] {param_name}: resized {list(value)} -> {trimmed} '
              f'for subbands {names}')
        value = trimmed
    return as_per_subband(value, names, param_name)


# ============================================================
# (d)/(e) SRST block with one branch removed
# ============================================================

class SRSTBlockAblate(nn.Module):
    """SRSTBlock, optionally without its STR branch (d) or spatial branch (e)."""

    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 k_spatial, groupnorm=True, groups=8,
                 use_str=True, use_spatial=True):
        super().__init__()
        assert use_str or use_spatial, 'cannot remove both SRST branches'
        self.use_str, self.use_spatial = use_str, use_spatial
        pad_spatial = (k_spatial - 1) // 2

        if use_str:
            self.str_branch = STRModule(dim, num_blocks, sparsity_threshold,
                                        hidden_size_factor=hidden_size_factor)
        if use_spatial:
            self.spatial_branch = nn.Conv2d(dim, dim, kernel_size=k_spatial,
                                            padding=pad_spatial, groups=dim,
                                            bias=False)
        self.norm = nn.GroupNorm(groups, dim) if groupnorm else nn.BatchNorm2d(dim)
        self.channel_mixing = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1), nn.GELU(), nn.Conv2d(dim * 2, dim, 1))
        self.act = nn.SiLU()

    def forward(self, x):
        x_ = x.permute(0, 3, 1, 2)                       # (B, D, H, W)
        parts = []
        if self.use_spatial:
            parts.append(self.spatial_branch(x_))
        if self.use_str:
            parts.append(self.str_branch(x_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x_fused = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        x_fused = self.channel_mixing(self.act(self.norm(x_fused)))
        return x_fused.permute(0, 2, 3, 1)               # (B, H, W, D)


class SRSTResBlockAblate(nn.Module):
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 k_spatial, groups=8, use_str=True, use_spatial=True):
        super().__init__()
        kw = dict(num_blocks=num_blocks, sparsity_threshold=sparsity_threshold,
                  hidden_size_factor=hidden_size_factor, k_spatial=k_spatial,
                  groups=groups, use_str=use_str, use_spatial=use_spatial)
        self.srst_block1 = SRSTBlockAblate(dim, **kw)
        self.srst_block2 = SRSTBlockAblate(dim, **kw)
        self.res_conv = nn.Identity()

    def forward(self, x):
        h = self.srst_block2(self.srst_block1(x))
        return h + self.res_conv(x)


# ============================================================
# (c) FAT block with the Gabor stream replaced by a Linear
# ============================================================

class FATBlockNoGabor(nn.Module):
    """
    FATBlock with the Gabor stream swapped for a single Linear(T_in, T_out).

    The dual-stream + fusion structure is preserved so the fusion Conv3d still
    receives 2*dim channels and the parameter count stays comparable.
    """

    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.gabor = nn.Linear(t_in, t_out)      # named 'gabor' to keep hooks/logging working
        self.mlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)), nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out))
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

    def forward(self, x):
        g = self.gabor(x)
        m = self.mlp(x)
        fused = torch.cat([g, m], dim=1).permute(0, 1, 4, 2, 3)
        return g, m, self.fusion(fused)


# ============================================================
# WGTM block with the wavelet transform optionally removed
# ============================================================

class WGTMBlockAblate(nn.Module):
    def __init__(self, t_in, t_out, dim, num_blocks, sparsity_threshold,
                 hidden_size_factor, weight_scale, alpha, beta, freq_multiplier,
                 k_spatial, size_factor=1.0, wave='db6', level=2,
                 hf_mode='separate', use_wavelet=True, use_gabor=True,
                 use_str=True, use_spatial=True, use_srst=True):
        super().__init__()
        self.t_in, self.t_out, self.dim = t_in, t_out, dim
        self.level, self.hf_mode = level, hf_mode
        self.use_wavelet, self.use_gabor = use_wavelet, use_gabor
        self.use_srst = use_srst

        names = ['LL'] if not use_wavelet else subband_names(level, hf_mode)
        self.subbands = names
        ws = _fit_per_subband(weight_scale, names, 'weight_scale')
        al = _fit_per_subband(alpha, names, 'alpha')
        be = _fit_per_subband(beta, names, 'beta')
        fm = _fit_per_subband(freq_multiplier, names, 'freq_multiplier')

        def make(idx, d):
            if use_gabor:
                return FATBlock(t_in, t_out, d, ws[idx], al[idx], be[idx],
                                fm[idx], size_factor)
            return FATBlockNoGabor(t_in, t_out, d, size_factor)

        if use_wavelet:
            self.dwt = DWTForward(J=level, wave=wave, mode='zero')
            self.idwt = DWTInverse(wave=wave, mode='zero')
            self.fat_ll = make(0, dim)
            if hf_mode == 'shared':
                self.fat_hf = make(1, 3 * dim)
            else:
                self.fat_hf_streams = nn.ModuleList(
                    [make(i + 1, 3 * dim) for i in range(level)])
        else:
            # (a) no wavelet: a single FAT block on the full-resolution features
            self.fat_full = make(0, dim)

        if use_srst:
            self.srst = nn.Sequential(
                SRSTResBlockAblate(dim * t_out, num_blocks, sparsity_threshold,
                                   hidden_size_factor, k_spatial,
                                   use_str=use_str, use_spatial=use_spatial),
                SRSTResBlockAblate(dim * t_out, num_blocks, sparsity_threshold,
                                   hidden_size_factor, k_spatial,
                                   use_str=use_str, use_spatial=use_spatial),
                STRModule(dim * t_out, num_blocks, sparsity_threshold,
                          hidden_size_factor=hidden_size_factor))

    def fat_blocks(self):
        out = OrderedDict()
        if not self.use_wavelet:
            out['FULL'] = self.fat_full
            return out
        out['LL'] = self.fat_ll
        if self.hf_mode == 'shared':
            out['HF_shared'] = self.fat_hf
        else:
            for i, b in enumerate(self.fat_hf_streams):
                out[f'HF_level_{i + 1}'] = b
        return out

    def gabor_layers(self):
        # `c_no_gabor` swaps the Gabor for a plain nn.Linear, which has no
        # freq/mu/gamma/freq_multiplier -- returning it here crashes the Gabor
        # logging. That ablation legitimately has no Gabor, so report none.
        if not self.use_gabor:
            return OrderedDict()
        return OrderedDict((n, b.gabor) for n, b in self.fat_blocks().items())

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')

        if not self.use_wavelet:
            xt = rearrange(x_flat, '(b t) c h w -> b c h w t', t=T)
            gabor_out, _, fused = self.fat_full(xt)
            recon = rearrange(fused, 'b c t h w -> (b t) c h w')
            residual = rearrange(gabor_out, 'b c h w t -> (b t) c h w')
        else:
            ll, hf_list = self.dwt(x_flat)
            ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
            ll_g, _, ll_f = self.fat_ll(ll_t)
            hf_g, hf_f = [], []
            for i, hf in enumerate(hf_list):
                hf_t = rearrange(hf, '(b t) c n h w -> b (c n) h w t', t=T)
                blk = self.fat_hf if self.hf_mode == 'shared' else self.fat_hf_streams[i]
                g, _, f = blk(hf_t)
                hf_g.append(g); hf_f.append(f)
            recon = self.idwt((rearrange(ll_f, 'b c t h w -> (b t) c h w'),
                               [rearrange(f, 'b (c n) t h w -> (b t) c n h w', n=3)
                                for f in hf_f]))
            residual = self.idwt((rearrange(ll_g, 'b c h w t -> (b t) c h w'),
                                  [rearrange(g, 'b (c n) h w t -> (b t) c n h w', n=3)
                                   for g in hf_g]))

        recon = rearrange(recon[..., :H, :W], '(b t) c h w -> b t c h w', t=self.t_out)
        residual = rearrange(residual[..., :H, :W], '(b t) c h w -> b t c h w', t=self.t_out)

        if not self.use_srst:
            # 'f_no_srst': refinement removed, Gabor residual bypass retained
            return recon + residual

        xs = rearrange(recon, 'b t c h w -> b h w (t c)')
        xs = self.srst(xs)
        xs = rearrange(xs, 'b h w (t c) -> b t c h w', t=self.t_out)
        return xs + residual


# ============================================================
# (g) whole WGTM replaced by a single temporal Linear
# ============================================================

class TemporalMLPOnly(nn.Module):
    """Lifting -> Linear(T_in, T_out) over the time axis -> Projection."""

    def __init__(self, t_in, t_out):
        super().__init__()
        self.proj = nn.Linear(t_in, t_out)

    def forward(self, x):                                  # (B, T_in, C, H, W)
        h = rearrange(x, 'b t c h w -> b c h w t')
        h = self.proj(h)
        return rearrange(h, 'b c h w t -> b t c h w')


# ============================================================
# Model + forecaster
# ============================================================

class DAWNCastAblation(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim, num_blocks,
                 sparsity_threshold, hidden_size_factor, weight_scale, alpha,
                 beta, freq_multiplier, k_spatial, size_factor=1.0, wave='db6',
                 level=2, hf_mode='separate', ablation='none'):
        super().__init__()
        assert ablation in ABLATIONS, f'unknown ablation {ablation}'
        self.ablation = ablation
        self.T_in, self.T_out = T_in, T_out

        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))

        if ablation == 'g_no_wgtm':
            self.wgtm = TemporalMLPOnly(T_in, T_out)
        else:
            self.wgtm = WGTMBlockAblate(
                T_in, T_out, hidden_dim, num_blocks, sparsity_threshold,
                hidden_size_factor, weight_scale, alpha, beta, freq_multiplier,
                k_spatial, size_factor, wave, level,
                hf_mode='shared' if ablation == 'b_shared_fat' else hf_mode,
                use_wavelet=(ablation != 'a_no_wavelet'),
                use_gabor=(ablation != 'c_no_gabor'),
                use_str=(ablation != 'd_no_str'),
                use_spatial=(ablation != 'e_no_spatial'),
                use_srst=(ablation != 'f_no_srst'))

        self.projection = nn.Sequential(
            TransformBlock(hidden_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1))

    def gabor_layers(self):
        return self.wgtm.gabor_layers() if hasattr(self.wgtm, 'gabor_layers') else OrderedDict()

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.lifting(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_in)
        x = self.wgtm(x)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.projection(x)
        return rearrange(x, '(b t) c h w -> b t c h w', t=self.T_out)


class DAWNCastAblationForecaster(nn.Module):
    def __init__(self, total_steps=50000, const_ratio=0.1, **kw):
        super().__init__()
        self.dawncast = DAWNCastAblation(**kw)
        self.T_in, self.T_out = kw['T_in'], kw['T_out']
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)

    def gabor_layers(self):
        return self.dawncast.gabor_layers()

    def forward(self, x, y=None, cmp_fft_loss=False):
        return self.dawncast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        pred = self.dawncast(frames_in)
        if not compute_loss:
            return pred, None
        facl = self.falfcl(pred, frames_gt)
        return pred, {'facl_loss': facl, 'total_loss': facl}


def get_model(ablation='none', T_in=5, T_out=20, img_channels=1, dim=64,
              afno_blocks=4, sparsity_threshold=0.01, afno_hidden_size_factor=4,
              weight_scale=1.0, alpha=1.0, beta=1.0, freq_multiplier=1.0,
              size_factor=1.0, total_steps=50000, const_ratio=0.1, k_spatial=3,
              wave='db6', wavelet_level=2, hf_mode='separate', **kwargs):
    return DAWNCastAblationForecaster(
        total_steps=total_steps, const_ratio=const_ratio,
        T_in=T_in, T_out=T_out, in_dim=img_channels, hidden_dim=dim,
        num_blocks=afno_blocks, sparsity_threshold=sparsity_threshold,
        hidden_size_factor=afno_hidden_size_factor, weight_scale=weight_scale,
        alpha=alpha, beta=beta, freq_multiplier=freq_multiplier,
        k_spatial=k_spatial, size_factor=size_factor, wave=wave,
        level=wavelet_level, hf_mode=hf_mode, ablation=ablation)
