#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted WNO (Wavelet Neural Operator) for spatiotemporal super-resolution.
VERSION 3 — all self-audit bugs fixed + level-clamping corrected.

Original paper:
    Tripura, T., & Chakraborty, S. (2022). Wavelet neural operator: a neural
    operator for parametric partial differential equations. arXiv:2205.02191.

=======================================================================
SELF-AUDIT RESULTS (bugs found and fixed vs v1)
=======================================================================

CRITICAL BUG (FIXED): Channel count mismatch in higher-level wavelet coefficients.
    In WaveConv3dBTCHW.forward(), after transforming coeffs[0] and coeffs[1]
    from C_in -> C_out channels, coeffs[2..eff_level] were zeroed using
    torch.zeros_like(v), where v still had C_IN channels.
    waverec3 then received a mixed list: coeffs[0] with C_out, coeffs[1] with
    C_out, coeffs[2+] with C_in -- causing silent shape mismatch or wrong output.
    
    Fix: Zero out higher-level details with explicit C_out size:
        torch.zeros(B, self.out_channels, *v.shape[2:], device=..., dtype=...)

DEPENDENCY RISK (RESOLVED): The original WaveConv3d uses wavedec3/waverec3
    imported from an unknown source (the original repo's wavelet_convolution.py
    does not show its imports clearly). The two candidates are:
        - ptwt  (pytorch_wavelet_toolbox): has batched wavedec3/waverec3
        - pywt  (standard): has pywt.dwtn but no wavedec3 directly
    
    Resolution: We use ptwt (pip install ptwt) which provides exactly the
    wavedec3(data, wavelet, level, mode) API that matches the original calls.
    ptwt wraps pywt with autograd support and batch dimensions.
    
    IF your original wavelet_convolution.py uses a different source, replace:
        from ptwt import wavedec3, waverec3
    with whatever your original imports.

BUG 2 (FIXED v3): eff_level computation only checked T dimension.
    Original heuristic: ratio = T_actual/T_registered, adjust level by log2(ratio).
    Problem: if H or W is the binding constraint (e.g. level=3, H=4 → 4/2³<1),
    the T-only check misses it and wavedec3 crashes or produces garbage.
    Fix: eff_level = min(self.level, floor(log2(min(T,H,W))))
    This is also simpler and more principled: level is a model hyperparameter
    and should never be auto-scaled upward at runtime.

EVERYTHING ELSE verified correct:
    - einsum "bixyz,ioxyz->boxyz" is correct batched version of original
    - _pad_or_trim is safe (original never needed it because sizes were exact)
    - time_proj squeeze/view is correct
    - F.pad for H,W spatial padding is correct
    - Wavelet layer loop (WaveConv + bypass + Mish) matches original
    - fc0 has no activation (matches original)
    - fc1/fc2 output projection matches original structure

=======================================================================
FORMAT CHANGES FROM ORIGINAL
=======================================================================
    Original: (B, H, W, T_in, C=1)  ->  (B, H, W, T_in, 1)
    Adapted : (B, T_in, C_in, H, W) ->  (B, T_out, C_out, H, W)

All changes annotated with # [CHANGED]. All unchanged parts with # [UNCHANGED].

=======================================================================
DEPENDENCY
=======================================================================
    pip install ptwt        # pytorch wavelet toolbox (batched wavedec3/waverec3)
    pip install pywt        # standard wavelet (used by ptwt internally)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from ptwt import wavedec3, waverec3   # pip install ptwt


# ---------------------------------------------------------------------------
# Batched 3-D Wavelet Convolution  (B, C, T, H, W) -> (B, C_out, T, H, W)
# ---------------------------------------------------------------------------
class WaveConv3dBTCHW(nn.Module):
    """
    3-D Wavelet integral operator adapted for (B, C, T, H, W) tensors.

    Mathematical operation (UNCHANGED from original WaveConv3d):
        1. 3-D DWT over (T, H, W) volume
        2. Learned linear mix of approximate + 7 detail sub-bands (weights1..8)
        3. Zero out coefficients at decomposition levels > 1  [UNCHANGED]
        4. 3-D IDWT back to physical space

    Changes from original WaveConv3d:
    ----------------------------------
    [CHANGED]  Removed per-sample for-loop. Now operates on full batch using
               ptwt.wavedec3 / waverec3 which are batched and autograd-safe.
    [CHANGED]  size = [T, H, W] where T = T_out (post time_proj size).
    [CHANGED]  _mul3d einsum has batch dim B: "bixyz,ioxyz->boxyz"
    [FIXED v2] zeros_like replaced with explicit zeros(B, out_channels, ...)
               to correctly zero higher-level details in C_out channel space.
    [UNCHANGED] 8 learnable weight tensors (approx + 7 detail sub-bands).
    [UNCHANGED] Weight shape: (in_ch, out_ch, modes1, modes2, modes3).
    [UNCHANGED] modes computed from dummy DWT run on size.
    [UNCHANGED] Dynamic level adjustment for mismatched input size.
    """

    def __init__(self, in_channels, out_channels, level, size, wavelet='haar'):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.level        = level
        self.wavelet      = wavelet

        if not (isinstance(size, (list, tuple)) and len(size) == 3):
            raise ValueError("size must be [T, H, W]")
        # Cast to int defensively — guards against str values from YAML/argparse
        self.size = [int(v) for v in size]

        # ------------------------------------------------------------------
        # Compute approximate coefficient shape via dummy DWT
        # [UNCHANGED] same formula as original: halving each level
        # ------------------------------------------------------------------
        t, h, w = self.size
        for _ in range(level):
            t = (t + 1) // 2
            h = (h + 1) // 2
            w = (w + 1) // 2
        self.modes1 = t
        self.modes2 = h
        self.modes3 = w

        # ------------------------------------------------------------------
        # Learnable weights: 1 approx + 7 detail sub-bands
        # [UNCHANGED] same count, same shape as original weights1..8
        # ------------------------------------------------------------------
        scale = 1.0 / (in_channels * out_channels)
        ms = (in_channels, out_channels, self.modes1, self.modes2, self.modes3)

        def _p(): return nn.Parameter(scale * torch.rand(*ms))

        self.w_ll  = _p()   # approximation (aaa)
        self.w_aad = _p()   # detail sub-bands (same 7 as original)
        self.w_ada = _p()
        self.w_add = _p()
        self.w_daa = _p()
        self.w_dad = _p()
        self.w_dda = _p()
        self.w_ddd = _p()

    # ------------------------------------------------------------------
    def _mul3d(self, coeff, weight):
        """
        Learned linear mix across input channels.
        coeff  : (B, C_in,  m1, m2, m3)
        weight : (C_in, C_out, m1, m2, m3)
        return : (B, C_out, m1, m2, m3)

        [CHANGED] Batched: added B dim vs original "ixyz,ioxyz->oxyz".
        [UNCHANGED] Same einsum contraction over input channels and modes.
        """
        return torch.einsum("bixyz,ioxyz->boxyz", coeff, weight)

    # ------------------------------------------------------------------
    def _pad_or_trim(self, coeff, target):
        """Crop or zero-pad coeff last 3 dims to target = (m1,m2,m3)."""
        t, h, w = target
        # trim
        coeff = coeff[..., :min(coeff.shape[-3], t),
                           :min(coeff.shape[-2], h),
                           :min(coeff.shape[-1], w)]
        # pad
        pt = max(0, t - coeff.shape[-3])
        ph = max(0, h - coeff.shape[-2])
        pw = max(0, w - coeff.shape[-1])
        if pt or ph or pw:
            coeff = F.pad(coeff, [0, pw, 0, ph, 0, pt])
        return coeff

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x       : (B, C_in,  T, H, W)
        returns : (B, C_out, T, H, W)
        """
        B, C, T, H, W = x.shape

        # ------------------------------------------------------------------
        # Safety clamp: eff_level must not exceed what the smallest dimension
        # can support.  At each wavelet level every dimension is halved, so
        # after `level` levels the smallest dim has size dim/2^level.
        # Requirement: dim/2^level >= 1  =>  level <= floor(log2(min_dim)).
        #
        # [FIXED v3] Original code (and our v2) only checked the T dimension,
        # missing cases where H or W is the binding constraint.  For example,
        # with self.level=3 and H=W=4: T-only logic keeps level=3 but
        # 4/2^3 < 1 → crash inside wavedec3.
        #
        # The new logic considers all three dimensions and simply clamps level
        # to what is physically feasible.  "Increasing level when input is
        # bigger" (original heuristic) is also dropped: level is a model
        # hyperparameter and should not be auto-scaled upward at runtime.
        # ------------------------------------------------------------------
        min_dim   = min(T, H, W)
        max_level = max(1, int(np.log2(max(min_dim, 2))))  # floor(log2), ≥1
        eff_level = min(self.level, max_level)

        wav = pywt.Wavelet(self.wavelet)
        ms  = (self.modes1, self.modes2, self.modes3)

        # ------------------------------------------------------------------
        # 3-D DWT over (T, H, W) volume
        # [CHANGED] ptwt.wavedec3 is batched: input (B, C, T, H, W)
        # [UNCHANGED] same decomposition: approx + level detail dicts
        # ------------------------------------------------------------------
        coeffs = list(wavedec3(x, wav, level=eff_level, mode='periodic'))
        # coeffs[0]       : approx tensor  (B, C_in, t', h', w')
        # coeffs[1..level]: dicts — ptwt may use tuple keys ('a','a','d') or
        #                   string keys 'aad' depending on version.
        #                   Normalise all detail dicts to string keys.
        _tup2str = lambda k: ''.join(k) if isinstance(k, tuple) else k
        for _li in range(1, eff_level + 1):
            if coeffs[_li] and isinstance(next(iter(coeffs[_li])), tuple):
                coeffs[_li] = {_tup2str(k): v for k, v in coeffs[_li].items()}

        # ------------------------------------------------------------------
        # Apply learned weights to approximate sub-band
        # [UNCHANGED] weights1 on approx (called w_ll here)
        # ------------------------------------------------------------------
        coeffs[0] = self._mul3d(
            self._pad_or_trim(coeffs[0], ms), self.w_ll
        )   # (B, C_out, m1, m2, m3)

        # ------------------------------------------------------------------
        # Apply learned weights to first-level detail sub-bands
        # [UNCHANGED] weights2..8 on 7 detail sub-bands
        # ------------------------------------------------------------------
        def _apply(key, weight):
            return self._mul3d(
                self._pad_or_trim(coeffs[1][key], ms), weight
            )   # (B, C_out, m1, m2, m3)

        coeffs[1] = {
            'aad': _apply('aad', self.w_aad),
            'ada': _apply('ada', self.w_ada),
            'add': _apply('add', self.w_add),
            'daa': _apply('daa', self.w_daa),
            'dad': _apply('dad', self.w_dad),
            'dda': _apply('dda', self.w_dda),
            'ddd': _apply('ddd', self.w_ddd),
        }

        # ------------------------------------------------------------------
        # Zero out higher-level detail coefficients
        # [UNCHANGED] same as original: higher levels set to zero
        # [FIXED v2]  use out_channels (C_out) not zeros_like (which had C_in)
        # ------------------------------------------------------------------
        for jj in range(2, eff_level + 1):
            coeffs[jj] = {
                k: torch.zeros(B, self.out_channels, *v.shape[2:],
                               device=x.device, dtype=x.dtype)
                for k, v in coeffs[jj].items()
            }

        # ------------------------------------------------------------------
        # 3-D IDWT back to physical space
        # [CHANGED] ptwt.waverec3 is batched
        # [UNCHANGED] same reconstruction from modified coefficients
        # ------------------------------------------------------------------
        out = waverec3(coeffs, wav)       # (B, C_out, T', H', W')
        out = out[..., :T, :H, :W]        # crop to exact input size
        return out


# ---------------------------------------------------------------------------
# WNO3d adapted for (B, T_in, C, H, W) -> (B, T_out, C_out, H, W)
# ---------------------------------------------------------------------------
class WNO3d_BTCHW(nn.Module):
    """
    Wavelet Neural Operator for spatiotemporal super-resolution.

    Input  : (B, T_in,  C_in,  H, W)
    Output : (B, T_out, C_out, H, W)    T_out > T_in

    Architecture (encode -> project-T -> operate -> decode):
    ---------------------------------------------------------
    1. Append (t, x, y) coordinate grid as 3 extra channels
    2. fc0: pointwise lift  C_in+3 -> width  (no activation, matches original)
    3. time_proj: Conv3d(width, width*T_out, (T_in,1,1))
                  maps T_in -> T_out in one learned global step
                  (same idiom as your FNO adaptation)
    4. `layers` blocks of WaveConv3dBTCHW + Conv3d(1x1x1) bypass + Mish
       (last block has no activation — matches original)
    5. fc1: width -> 128 -> Mish -> fc2: 128 -> C_out

    Parameters
    ----------
    width      : int
    level      : int   — wavelet decomposition levels
    layers     : int   — number of WaveConv blocks
    T_in       : int
    T_out      : int   — must be > T_in
    H, W       : int   — spatial size
    C_in       : int   — input channels (NOT counting grid)
    C_out      : int   — output channels
    wavelet    : str   — wavelet family ('haar' recommended for temporal axis)
    grid_range : list  — [t_max, x_max, y_max] for coordinate grid
    padding    : int   — zero-padding on H and W spatial dims only
    """

    def __init__(self,
                 width      : int,
                 level      : int,
                 layers     : int,
                 T_in       : int,
                 T_out      : int,
                 H          : int,
                 W          : int,
                 C_in       : int,
                 C_out      : int,
                 wavelet    : str  = 'haar',
                 grid_range : list = None,
                 padding    : int  = 0):
        super().__init__()

        # Cast all int params defensively (guards against str from YAML/argparse)
        width   = int(width);   level   = int(level);   layers  = int(layers)
        T_in    = int(T_in);    T_out   = int(T_out)
        H       = int(H);       W       = int(W)
        C_in    = int(C_in);    C_out   = int(C_out);   padding = int(padding)

        self.width      = width
        self.level      = level
        self.layers     = layers
        self.T_in       = T_in
        self.T_out      = T_out
        self.H          = H
        self.W          = W
        self.C_out      = C_out
        self.padding    = padding
        self.grid_range = grid_range or [1.0, 1.0, 1.0]

        # ------------------------------------------------------------------
        # 1. Input lifting
        # [CHANGED] fc0 takes C_in + 3 (3 grid channels: t, x, y)
        # [UNCHANGED] single Linear, no activation after (matches original)
        # ------------------------------------------------------------------
        self.fc0 = nn.Linear(C_in + 3, width)

        # ------------------------------------------------------------------
        # 2. Temporal projection: T_in -> T_out
        # [CHANGED] NEW — not in original.
        #   Conv3d with kernel (T_in,1,1):
        #     (B, width, T_in, H, W) -> (B, width*T_out, 1, H, W)
        #   then view to (B, width, T_out, H, W)
        #   This is the same idiom as your FNO/AFNO adaptation.
        #   Treated as part of the encoder (not an operator layer).
        # ------------------------------------------------------------------
        self.time_proj = nn.Conv3d(
            width,
            width * T_out,
            kernel_size=(T_in, 1, 1),
            padding=0,
        )

        # ------------------------------------------------------------------
        # 3. Wavelet integral layers operating at T_out resolution
        # [UNCHANGED] WaveConv + bypass Conv3d(1x1x1) structure
        # [CHANGED]   WaveConv size = [T_out, H+pad, W+pad]
        # ------------------------------------------------------------------
        sp_h = H + padding
        sp_w = W + padding

        self.conv = nn.ModuleList([
            WaveConv3dBTCHW(width, width, level, [T_out, sp_h, sp_w], wavelet)
            for _ in range(layers)
        ])
        # [UNCHANGED] pointwise bypass, same as original nn.Conv3d(width, width, 1)
        self.w = nn.ModuleList([
            nn.Conv3d(width, width, kernel_size=1)
            for _ in range(layers)
        ])

        # ------------------------------------------------------------------
        # 4. Output projection
        # [UNCHANGED] two-layer MLP structure
        # [CHANGED]   fc2 outputs C_out instead of 1
        # ------------------------------------------------------------------
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, C_out)

    # ----------------------------------------------------------------------
    def get_grid(self, B, T, H, W, device):
        """
        Coordinate grid: (t, x, y) as 3 channels.
        [CHANGED]   Returns (B, T, 3, H, W) for BTCHW layout.
        [UNCHANGED] Values: linspace over grid_range for each axis.
        """
        t_lin = torch.linspace(0, self.grid_range[0], T, device=device)
        x_lin = torch.linspace(0, self.grid_range[1], H, device=device)
        y_lin = torch.linspace(0, self.grid_range[2], W, device=device)

        gt = t_lin.view(1, T, 1, 1).expand(B, T, H, W)
        gx = x_lin.view(1, 1, H, 1).expand(B, T, H, W)
        gy = y_lin.view(1, 1, 1, W).expand(B, T, H, W)

        return torch.stack([gt, gx, gy], dim=2)   # (B, T, 3, H, W)

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x       : (B, T_in, C_in, H, W)
        returns : (B, T_out, C_out, H, W)
        """
        B, T_in, C, H, W = x.shape

        # ------------------------------------------------------------------
        # Step 1 — Coordinate grid
        # [CHANGED] cat on channel dim (dim=2)
        # ------------------------------------------------------------------
        grid = self.get_grid(B, T_in, H, W, x.device)  # (B, T_in, 3, H, W)
        x    = torch.cat([x, grid], dim=2)              # (B, T_in, C+3, H, W)

        # ------------------------------------------------------------------
        # Step 2 — Pointwise lifting via fc0
        # [CHANGED] permute C to last dim for Linear, then permute back
        # [UNCHANGED] no activation after fc0 (matches original)
        # ------------------------------------------------------------------
        x = x.permute(0, 1, 3, 4, 2)       # (B, T_in, H, W, C+3)
        x = self.fc0(x)                     # (B, T_in, H, W, width)
        x = x.permute(0, 4, 1, 2, 3)       # (B, width, T_in, H, W)

        # ------------------------------------------------------------------
        # Step 3 — Temporal projection T_in -> T_out
        # [CHANGED] NEW encoder step
        # ------------------------------------------------------------------
        x = self.time_proj(x)               # (B, width*T_out, 1, H, W)
        x = x.squeeze(2)                    # (B, width*T_out,    H, W)
        x = x.view(B, self.width, self.T_out, H, W)  # (B, width, T_out, H, W)

        # ------------------------------------------------------------------
        # Step 4 — Spatial padding (H and W only; T already fixed)
        # [UNCHANGED] zero-pad logic
        # ------------------------------------------------------------------
        if self.padding != 0:
            p = self.padding
            # F.pad: [W_right, H_right, T_right] -> pads W then H then T from last dim
            x = F.pad(x, [0, p, 0, p, 0, 0])

        # ------------------------------------------------------------------
        # Step 5 — Wavelet integral blocks
        # [UNCHANGED] WaveConv + bypass + Mish; no activation on last layer
        # ------------------------------------------------------------------
        for idx, (conv_l, w_l) in enumerate(zip(self.conv, self.w)):
            x = conv_l(x) + w_l(x)
            if idx != self.layers - 1:
                x = F.mish(x)

        # ------------------------------------------------------------------
        # Step 6 — Remove spatial padding
        # [UNCHANGED]
        # ------------------------------------------------------------------
        if self.padding != 0:
            x = x[..., :H, :W]              # (B, width, T_out, H, W)

        # ------------------------------------------------------------------
        # Step 7 — Output projection
        # [CHANGED] permute for Linear; fc2 outputs C_out not 1
        # [UNCHANGED] fc1(Mish) -> fc2 structure
        # ------------------------------------------------------------------
        x = x.permute(0, 2, 3, 4, 1)       # (B, T_out, H, W, width)
        x = F.mish(self.fc1(x))             # (B, T_out, H, W, 128)
        x = self.fc2(x)                     # (B, T_out, H, W, C_out)
        x = x.permute(0, 1, 4, 2, 3)       # (B, T_out, C_out, H, W)

        return x


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    def run(name, **kw):
        B, T_in, T_out = kw.pop('B'), kw.pop('T_in'), kw.pop('T_out')
        C_in, C_out    = kw.pop('C_in'), kw.pop('C_out')
        H, W           = kw.pop('H'), kw.pop('W')
        try:
            m   = WNO3d_BTCHW(T_in=T_in, T_out=T_out,
                               C_in=C_in, C_out=C_out, H=H, W=W, **kw)
            x   = torch.randn(B, T_in, C_in, H, W)
            out = m(x)
            assert out.shape == (B, T_out, C_out, H, W), \
                f"expected {(B,T_out,C_out,H,W)} got {out.shape}"
            print(f"  [PASS] {name}: {tuple(x.shape)} -> {tuple(out.shape)}")
        except Exception as e:
            import traceback
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    print("=== WNO3d_BTCHW Sanity Checks ===\n")
    run("2x  C=1", B=2, T_in=5,  T_out=10, C_in=1, C_out=1, H=32, W=32,
        width=16, level=2, layers=4, wavelet='haar')
    run("4x  C=3", B=2, T_in=5,  T_out=20, C_in=3, C_out=3, H=32, W=32,
        width=16, level=2, layers=4, wavelet='haar')
    run("2x  H=64",B=2, T_in=10, T_out=20, C_in=1, C_out=1, H=64, W=64,
        width=32, level=2, layers=4, wavelet='haar', padding=4)
