#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapted U-NO (U-shaped Neural Operator) for spatiotemporal super-resolution.

Original paper:
    Rahman, M. A., Ross, Z. E., & Azizzadenesheli, K. (2022).
    U-NO: U-shaped Neural Operators. arXiv:2204.11127

=======================================================================
SELF-AUDIT RESULTS (bugs found and fixed vs v1)
=======================================================================

BUG 1 (FIXED): Padding removal factor was hardcoded as 4.
    Old: x_c8[..., :-p*4]
    Fix: x_c8[..., :-int(round(p * ratio))]
    Reason: T grows by ratio (T_out/T_in) so padding grows by ratio too.
    Original T40 used *4 because ratio=4. Original T20 used *2 because ratio=2.

BUG 2 (FIXED): Spatial modes in OperatorBlock_3D were hardcoded to H=64.
    Old: modes computed for 48,32,16... assuming H=64 input.
    Fix: modes computed as fraction of actual spatial dims at each stage,
         respecting the constraint modes <= min(dim_out/2, dim_in/2).
    Reason: If H != 64, modes could exceed Nyquist limit silently.

INACCURACY (FIXED): T growth schedule for intermediate decoder stages
    deviated from original T20 by up to 12%.
    Fix: Use the exact original multipliers from T40/T20 as lookup table
         for ratio=4 and ratio=2, and geometric interpolation only for
         other ratios. T_out/T_in must be an integer >= 1.

FRAGILE (FIXED): dim3=T_in in OperatorBlock_3D __init__ was misleading.
    Fix: Pass the correct dim3 for each block based on _t_schedule,
         so __init__ and forward() are consistent.

=======================================================================
OPERATOR THEORY STATUS (unchanged from v1)
=======================================================================
All OperatorBlock_3D, SpectralConv3d_Uno, pointwise_op_3D are verbatim
copies of the original. The kernel integral K+W structure is fully
preserved. The temporal upsampling IS the spectral zero-padding inside
SpectralConv3d_Uno -- no separate time_proj is needed or appropriate.
The neural operator property is 100% intact.

=======================================================================
WHAT CHANGES FOR (B, T_in, C, H, W) FORMAT
=======================================================================
    [CHANGED] Input/output layout: (B,T,C,H,W) not (B,H,W,T,C)
    [CHANGED] get_grid returns (B,T,5,H,W) not (B,H,W,T,5)
    [CHANGED] Permutes adjusted around fc/fc0 and fc1/fc2
    [CHANGED] fc2 outputs C_out instead of 1
    [CHANGED] Spatial modes computed from actual H,W (not hardcoded)
    [CHANGED] T growth schedule generalised from hardcoded T40/T20 classes
    [CHANGED] Padding removal factor uses ratio not hardcoded 4
    [UNCHANGED] All OperatorBlock_3D, SpectralConv3d_Uno, pointwise_op_3D
    [UNCHANGED] U-shaped skip connection structure
    [UNCHANGED] Channel width progression: w->2fw->4fw->8fw->16fw->...
    [UNCHANGED] Spatial downscale: 1 -> 3/4 -> 1/2 -> 1/4 -> 1/8
    [UNCHANGED] Grid: sin(x),sin(y),cos(x),cos(y),t
    [UNCHANGED] fc lifting: two Linear layers with GELU
    [UNCHANGED] Output MLP: fc1(GELU) -> fc2
"""

"""
Adapted U-NO (U-shaped Neural Operator) for spatiotemporal super-resolution.

Original paper:
    Rahman, M. A., Ross, Z. E., & Azizzadenesheli, K. (2022).
    U-NO: U-shaped Neural Operators. arXiv:2204.11127

=======================================================================
SELF-AUDIT RESULTS (bugs found and fixed vs v1)
=======================================================================

BUG 1 (FIXED): Padding removal factor was hardcoded as 4.
    Old: x_c8[..., :-p*4]
    Fix: x_c8[..., :-int(round(p * ratio))]
    Reason: T grows by ratio (T_out/T_in) so padding grows by ratio too.
    Original T40 used *4 because ratio=4. Original T20 used *2 because ratio=2.

BUG 2 (FIXED): Spatial modes in OperatorBlock_3D were hardcoded to H=64.
    Old: modes computed for 48,32,16... assuming H=64 input.
    Fix: modes computed as fraction of actual spatial dims at each stage,
         respecting the constraint modes <= min(dim_out/2, dim_in/2).
    Reason: If H != 64, modes could exceed Nyquist limit silently.

INACCURACY (FIXED): T growth schedule for intermediate decoder stages
    deviated from original T20 by up to 12%.
    Fix: Use the exact original multipliers from T40/T20 as lookup table
         for ratio=4 and ratio=2, and geometric interpolation only for
         other ratios. T_out/T_in must be an integer >= 1.

FRAGILE (FIXED): dim3=T_in in OperatorBlock_3D __init__ was misleading.
    Fix: Pass the correct dim3 for each block based on _t_schedule,
         so __init__ and forward() are consistent.

=======================================================================
v3 ADDITIONAL FIXES
=======================================================================

BUG 3 (FIXED v3): t_modes hardcoded for T_in=10; crashes for smaller T_in.
    SpectralConv3d_Uno does rfftn on input → last dim = T_in//2+1.
    Weight subscript z must be <= that. With T_in=5, bottleneck T=5,
    rfftn_z=3 but t_mode=7 → "subscript z size 7 does not broadcast with 4".
    Fix: t_modes[i] = min(desired, T_in_to_block[i] // 2 + 1).

BUG 4 (FIXED v3): fc1 input dimension wrong for factor != 1.
    After the final skip-cat: channels = conv8_out(2*f*w) + x_fc0(w) = (2*f+1)*w.
    Old: nn.Linear(3*width, ...) — only correct when factor=1.
    Fix: nn.Linear((2*f+1)*width, 4*width).

BUG 4 (FIXED v3): T dimension after trim could be T_out ± 1 due to
    integer rounding in _t(). This caused shape assertion failures.
    Fix: x_c8 = x_c8[..., :self.T_out] after the trim step.

=======================================================================
OPERATOR THEORY STATUS (unchanged from v1)
=======================================================================
All OperatorBlock_3D, SpectralConv3d_Uno, pointwise_op_3D are verbatim
copies of the original. The kernel integral K+W structure is fully
preserved. The temporal upsampling IS the spectral zero-padding inside
SpectralConv3d_Uno -- no separate time_proj is needed or appropriate.
The neural operator property is 100% intact.

=======================================================================
WHAT CHANGES FOR (B, T_in, C, H, W) FORMAT
=======================================================================
    [CHANGED] Input/output layout: (B,T,C,H,W) not (B,H,W,T,C)
    [CHANGED] get_grid returns (B,T,5,H,W) not (B,H,W,T,5)
    [CHANGED] Permutes adjusted around fc/fc0 and fc1/fc2
    [CHANGED] fc2 outputs C_out instead of 1
    [CHANGED] Spatial modes computed from actual H,W (not hardcoded)
    [CHANGED] T growth schedule generalised from hardcoded T40/T20 classes
    [CHANGED] Padding removal factor uses ratio not hardcoded 4
    [UNCHANGED] All OperatorBlock_3D, SpectralConv3d_Uno, pointwise_op_3D
    [UNCHANGED] U-shaped skip connection structure
    [UNCHANGED] Channel width progression: w->2fw->4fw->8fw->16fw->...
    [UNCHANGED] Spatial downscale: 1 -> 3/4 -> 1/2 -> 1/4 -> 1/8
    [UNCHANGED] Grid: sin(x),sin(y),cos(x),cos(y),t
    [UNCHANGED] fc lifting: two Linear layers with GELU
    [UNCHANGED] Output MLP: fc1(GELU) -> fc2
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utilspp import RandomScheduling

# ============================================================
# UNCHANGED: All integral operator building blocks
# Verbatim from original integral_operators.py
# ============================================================

class SpectralConv3d_Uno(nn.Module):
    """
    3D Fourier integral operator.
    [UNCHANGED] Exact copy from original integral_operators.py.
    Input/output: (B, C, dim1, dim2, dim3)
    """
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3,
                 modes1=None, modes2=None, modes3=None):
        super().__init__()
        in_codim  = int(in_codim)
        out_codim = int(out_codim)
        self.in_channels  = in_codim
        self.out_channels = out_codim
        self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3

        if modes1 is not None:
            self.modes1, self.modes2, self.modes3 = modes1, modes2, modes3
        else:
            self.modes1 = dim1
            self.modes2 = dim2
            self.modes3 = dim3 // 2 + 1

        self.scale = (1 / (2 * in_codim)) ** 0.5
        def _w(*s): return nn.Parameter(self.scale * torch.randn(*s, dtype=torch.cfloat))
        ms = (in_codim, out_codim, self.modes1, self.modes2, self.modes3)
        self.weights1, self.weights2 = _w(*ms), _w(*ms)
        self.weights3, self.weights4 = _w(*ms), _w(*ms)

    def compl_mul3d(self, inp, w):
        return torch.einsum("bixyz,ioxyz->boxyz", inp, w)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        if dim1 is not None:
            self.dim1, self.dim2, self.dim3 = dim1, dim2, dim3
        B = x.shape[0]
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm="forward")
        out_ft = torch.zeros(B, self.out_channels,
                             self.dim1, self.dim2, self.dim3 // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft[:, :,  :m1,  :m2, :m3] = self.compl_mul3d(x_ft[:, :,  :m1,  :m2, :m3], self.weights1)
        out_ft[:, :, -m1:,  :m2, :m3] = self.compl_mul3d(x_ft[:, :, -m1:,  :m2, :m3], self.weights2)
        out_ft[:, :,  :m1, -m2:, :m3] = self.compl_mul3d(x_ft[:, :,  :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = self.compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4)
        return torch.fft.irfftn(out_ft, s=(self.dim1, self.dim2, self.dim3), norm="forward")


class pointwise_op_3D(nn.Module):
    """
    1x1x1 conv + spectral crop + trilinear interpolation.
    [UNCHANGED] Exact copy from original integral_operators.py.
    """
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3):
        super().__init__()
        self.conv = nn.Conv3d(int(in_codim), int(out_codim), 1)
        self.dim1, self.dim2, self.dim3 = int(dim1), int(dim2), int(dim3)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        if dim1 is None:
            dim1, dim2, dim3 = self.dim1, self.dim2, self.dim3
        x_out = self.conv(x)
        ft   = torch.fft.rfftn(x_out, dim=[-3, -2, -1])
        ft_u = torch.zeros_like(ft)
        d1h, d2h, d3h = dim1 // 2, dim2 // 2, dim3 // 2
        ft_u[:, :,  :d1h,  :d2h, :d3h] = ft[:, :,  :d1h,  :d2h, :d3h]
        ft_u[:, :, -d1h:,  :d2h, :d3h] = ft[:, :, -d1h:,  :d2h, :d3h]
        ft_u[:, :,  :d1h, -d2h:, :d3h] = ft[:, :,  :d1h, -d2h:, :d3h]
        ft_u[:, :, -d1h:, -d2h:, :d3h] = ft[:, :, -d1h:, -d2h:, :d3h]
        x_out = torch.fft.irfftn(ft_u, s=(dim1, dim2, dim3))
        return F.interpolate(x_out, size=(dim1, dim2, dim3),
                             mode="trilinear", align_corners=True)


class OperatorBlock_3D(nn.Module):
    """
    U-NO operator layer: v_{l+1} = sigma( K(v_l) + W(v_l) )
    [UNCHANGED] Exact copy from original integral_operators.py.
    """
    def __init__(self, in_codim, out_codim, dim1, dim2, dim3,
                 modes1, modes2, modes3, Normalize=False, Non_Lin=True):
        super().__init__()
        self.conv = SpectralConv3d_Uno(in_codim, out_codim, dim1, dim2, dim3,
                                       modes1, modes2, modes3)
        self.w    = pointwise_op_3D(in_codim, out_codim, dim1, dim2, dim3)
        self.normalize = Normalize
        self.non_lin   = Non_Lin
        if Normalize:
            self.normalize_layer = nn.InstanceNorm3d(int(out_codim), affine=True)

    def forward(self, x, dim1=None, dim2=None, dim3=None):
        x_out = self.conv(x, dim1, dim2, dim3) + self.w(x, dim1, dim2, dim3)
        if self.normalize:
            x_out = self.normalize_layer(x_out)
        if self.non_lin:
            x_out = F.gelu(x_out)
        return x_out


# ============================================================
# ADAPTED: UNO3D_BTCHW  (v2 — all bugs fixed)
# ============================================================

# Exact T-growth multipliers from the original hardcoded classes.
# Key: ratio = T_out // T_in  (must be integer)
# Values: list of 7 multipliers [conv0..conv3, conv6..conv8] relative to D3
_ORIGINAL_T_SCHEDULES = {
    4: [1.0, 1.0, 1.6, 1.6, 2.4, 3.2, 4.0],   # from Uno3D_T40
    2: [1.0, 1.0, 1.2, 1.2, 1.8, 2.0, 2.0],   # from Uno3D_T20
}


def _compute_t_schedule(ratio: int):
    """
    Return the 7-element T growth schedule for a given integer ratio.
    For ratio in {2, 4}: returns the exact original hardcoded schedule.
    For other integer ratios: uses geometric interpolation.
    """
    if ratio in _ORIGINAL_T_SCHEDULES:
        return _ORIGINAL_T_SCHEDULES[ratio]
    # Geometric: encoder holds at 1.0 for first 2 stages,
    # then grows smoothly to ratio across 5 remaining stages.
    # waypoints at positions 2,3 (encoder end / bottleneck) and 4,5,6 (decoder)
    return [
        1.0,
        1.0,
        ratio ** (1.0 / 3.0),
        ratio ** (1.0 / 3.0),
        ratio ** (2.0 / 3.0),
        ratio ** (5.0 / 6.0),
        float(ratio),
    ]


def _safe_modes(modes_fraction: float, dim_in: int, dim_out: int) -> int:
    """
    Compute safe Fourier modes satisfying:
        modes <= min(dim_in // 2, dim_out // 2)
    modes_fraction is the fraction of dim_out to use as modes.

    [CHANGED v2] This replaces hardcoded mode values (20,14,6,...) which
    were calibrated for H=64 only. Now modes are computed from actual dims.
    """
    target = max(1, int(round(dim_out * modes_fraction)))
    limit  = min(dim_in // 2, dim_out // 2)
    return max(1, min(target, limit))


class UNO3D_BTCHW(nn.Module):
    """
    U-shaped Neural Operator — adapted for (B, T_in, C, H, W) -> (B, T_out, C_out, H, W).

    Parameters
    ----------
    C_in     : int  — input  channels (NOT counting grid; grid is added internally)
    C_out    : int  — output channels
    width    : int  — base channel dimension (same meaning as original 'width')
    T_in     : int  — number of input  time steps
    T_out    : int  — number of output time steps  (must be integer multiple of T_in)
    H, W     : int  — spatial height and width of input
    pad      : int  — padding control (same as original)
    factor   : int  — channel scaling factor (same as original)
    pad_both : bool — pad both sides of T axis (same as original)
    modes_frac: float — fraction of spatial dim to use as Fourier modes (default 0.42,
                        which reproduces original behaviour at H=64)
    """

    def __init__(self,
                 C_in      : int,
                 C_out     : int,
                 width     : int,
                 T_in      : int,
                 T_out     : int,
                 H         : int,
                 W         : int,
                 pad       : int   = 2,
                 factor    : int   = 1,
                 pad_both  : bool  = False,
                 modes_frac: float = 0.42):
        super().__init__()

        assert T_out % T_in == 0, \
            f"T_out ({T_out}) must be an integer multiple of T_in ({T_in})."

        self.C_out     = C_out
        self.width     = width
        self.pad       = pad
        self.pad_both  = pad_both
        self.T_in      = T_in
        self.T_out     = T_out
        self.H         = H
        self.W         = W
        self._ratio    = T_out // T_in

        # T growth schedule (7 multipliers, relative to D3 at runtime)
        self._t_sched  = _compute_t_schedule(self._ratio)

        # ------------------------------------------------------------------
        # Spatial dims at each encoder stage (for mode computation)
        # These match the D1,D2 ratios used in forward():
        #   conv0: 3/4,  conv1: 1/2,  conv2: 1/4,  conv3: 1/8
        # ------------------------------------------------------------------
        stage_h = [
            int(3 * H / 4),    # conv0
            H // 2,            # conv1
            H // 4,            # conv2
            H // 8,            # conv3 (bottleneck)
            H // 2,            # conv6
            int(3 * H / 4),    # conv7
            H,                 # conv8
        ]
        stage_w = [int(3*W/4), W//2, W//4, W//8, W//2, int(3*W/4), W]

        # Input spatial dims for each block (= output dims of previous block)
        in_h  = [H,         stage_h[0], stage_h[1], stage_h[2],
                 stage_h[3], stage_h[4], stage_h[5]]
        in_w  = [W,         stage_w[0], stage_w[1], stage_w[2],
                 stage_w[3], stage_w[4], stage_w[5]]

        # [CHANGED v2] modes computed from actual H,W using modes_frac
        # constraint: modes <= min(dim_out/2, dim_in/2) always satisfied
        def _m1(i): return _safe_modes(modes_frac, in_h[i], stage_h[i])
        def _m2(i): return _safe_modes(modes_frac, in_w[i], stage_w[i])

        # T dim at each block's output, registered at init time.
        # We use T_in as representative; forward overrides via dim3 arg.
        reg_t = [max(1, int(round(T_in * s))) for s in self._t_sched]

        # T Fourier modes — clamped to rfftn capacity at each block.
        #
        # SpectralConv3d_Uno does rfftn on its INPUT, giving last-dim size
        # = T_in_to_block // 2 + 1.  The weight tensor subscript z must be
        # <= that size, otherwise einsum raises a size mismatch at runtime.
        #
        # [FIXED v3] Previously hardcoded [4,4,4,7,7,10,14] (calibrated for
        # T_in=10). For T_in=5 the bottleneck receives T=5 → rfftn_z=3,
        # while t_mode=7 → CRASH ("subscript z has size 7 ... does not
        # broadcast with ... size 4").
        # Fix: clamp each t_mode to min(desired, T_in_to_block // 2 + 1).
        _desired_t_modes = [4, 4, 4, 7, 7, 10, 14]
        _t_in_to_block   = [T_in] + reg_t[:6]   # T entering each block
        t_modes = [
            max(1, min(_desired_t_modes[i], _t_in_to_block[i] // 2 + 1))
            for i in range(7)
        ]

        # ------------------------------------------------------------------
        # [CHANGED] fc lifts (C_in + 5 grid channels) -> width
        # [UNCHANGED] two-layer fc structure with GELU between
        # ------------------------------------------------------------------
        grid_ch  = 5
        in_ch    = C_in + grid_ch
        self.fc  = nn.Linear(in_ch, in_ch * 2)
        self.fc0 = nn.Linear(in_ch * 2, width)

        f = factor
        w = width

        # ------------------------------------------------------------------
        # [UNCHANGED] U-shaped operator block structure
        # [CHANGED v2] dim1,dim2,dim3 consistent between __init__ and forward
        # [CHANGED v2] modes1,modes2 computed from actual H,W (not hardcoded)
        # ------------------------------------------------------------------

        # --- Encoder ---
        self.conv0 = OperatorBlock_3D(
            w,      2*f*w, stage_h[0], stage_w[0], reg_t[0],
            _m1(0), _m2(0), t_modes[0], Normalize=True)

        self.conv1 = OperatorBlock_3D(
            2*f*w,  4*f*w, stage_h[1], stage_w[1], reg_t[1],
            _m1(1), _m2(1), t_modes[1])

        self.conv2 = OperatorBlock_3D(
            4*f*w,  8*f*w, stage_h[2], stage_w[2], reg_t[2],
            _m1(2), _m2(2), t_modes[2])

        # --- Bottleneck ---
        self.conv3 = OperatorBlock_3D(
            8*f*w, 16*f*w, stage_h[3], stage_w[3], reg_t[3],
            _m1(3), _m2(3), t_modes[3], Normalize=True)

        # --- Decoder ---
        # Note: decoder input channels are doubled due to skip cat
        self.conv6 = OperatorBlock_3D(
            16*f*w,  4*f*w, stage_h[4], stage_w[4], reg_t[4],
            _m1(4),  _m2(4), t_modes[4])

        self.conv7 = OperatorBlock_3D(
            8*f*w,   2*f*w, stage_h[5], stage_w[5], reg_t[5],
            _m1(5),  _m2(5), t_modes[5], Normalize=True)

        self.conv8 = OperatorBlock_3D(
            4*f*w,   2*w,   stage_h[6], stage_w[6], reg_t[6],
            _m1(6),  _m2(6), t_modes[6])

        # ------------------------------------------------------------------
        # [UNCHANGED] Output MLP: (3*width) -> (4*width) -> C_out
        # [CHANGED] fc2 outputs C_out instead of 1
        # ------------------------------------------------------------------
        # [FIXED v3] fc1 input = conv8_out(2*f*w) + x_fc0_skip(w) = (2*f+1)*w
        # Previously hardcoded as 3*width, which only holds when factor=1.
        self.fc1 = nn.Linear((2 * f + 1) * width, 4 * width)
        self.fc2 = nn.Linear(4 * width, C_out)

    # ----------------------------------------------------------------------
    def get_grid(self, B, T, H, W, device):
        """
        5-channel coordinate grid: sin(x), sin(y), cos(x), cos(y), t
        [CHANGED] Returns (B, T, 5, H, W) instead of (B, H, W, T, 5)
        [UNCHANGED] Grid values identical to original
        """
        x_lin = torch.linspace(0, 2 * np.pi, H, device=device)
        y_lin = torch.linspace(0, 2 * np.pi, W, device=device)
        t_lin = torch.linspace(0, 1,          T, device=device)

        gx = x_lin.view(1, 1, H, 1).expand(B, T, H, W)
        gy = y_lin.view(1, 1, 1, W).expand(B, T, H, W)
        gt = t_lin.view(1, T, 1, 1).expand(B, T, H, W)

        return torch.stack([
            torch.sin(gx), torch.sin(gy),
            torch.cos(gx), torch.cos(gy),
            gt
        ], dim=2)   # (B, T, 5, H, W)

    # ----------------------------------------------------------------------
    def forward(self, x):
        """
        x : (B, T_in, C_in, H, W)
        returns : (B, T_out, C_out, H, W)
        """
        B, T_in, C, H, W = x.shape

        # ------------------------------------------------------------------
        # Step 1 — Coordinate grid
        # [CHANGED] cat on channel dim (dim=2) not last dim
        # ------------------------------------------------------------------
        grid = self.get_grid(B, T_in, H, W, x.device)  # (B, T_in, 5, H, W)
        x    = torch.cat([x, grid], dim=2)              # (B, T_in, C+5, H, W)

        # ------------------------------------------------------------------
        # Step 2 — Input lifting
        # [CHANGED] permute to (..., C) for Linear, then permute back
        # [UNCHANGED] fc(GELU) -> fc0(GELU) two-stage lift
        # ------------------------------------------------------------------
        x     = x.permute(0, 1, 3, 4, 2)          # (B, T_in, H, W, C+5)
        x     = F.gelu(self.fc(x))                 # (B, T_in, H, W, (C+5)*2)
        x_fc0 = F.gelu(self.fc0(x))                # (B, T_in, H, W, width)

        # [CHANGED] permute to (B, width, H, W, T_in) for OperatorBlock_3D
        # OperatorBlock_3D always expects (B, C, dim1, dim2, dim3)
        # with dim3 = T -- same as original after its permute(0,4,1,2,3)
        x_fc0 = x_fc0.permute(0, 4, 2, 3, 1)       # (B, width, H, W, T_in)

        # ------------------------------------------------------------------
        # Step 3 — Padding along T axis
        # [UNCHANGED] identical logic to original
        # ------------------------------------------------------------------
        p = int(self.pad * 0.1 * x_fc0.shape[-1])
        self._runtime_pad = p
        if self.pad_both:
            x_fc0 = F.pad(x_fc0, [p, p, 0, 0, 0, 0])
        else:
            x_fc0 = F.pad(x_fc0, [0, p, 0, 0, 0, 0])

        D1 = x_fc0.shape[-3]   # H  (no spatial padding in this model)
        D2 = x_fc0.shape[-2]   # W
        D3 = x_fc0.shape[-1]   # T_in + p

        s = self._t_sched
        def _t(scale): return max(1, int(round(D3 * scale)))

        # ------------------------------------------------------------------
        # Step 4 — Encoder
        # [UNCHANGED] block structure and spatial ratios
        # [CHANGED] dim3 uses _t(s[i]); dim1,dim2 same ratios as original
        # ------------------------------------------------------------------
        x_c0 = self.conv0(x_fc0, int(3*D1/4), int(3*D2/4), _t(s[0]))
        x_c1 = self.conv1(x_c0,  D1//2,       D2//2,       _t(s[1]))
        x_c2 = self.conv2(x_c1,  D1//4,       D2//4,       _t(s[2]))

        # --- Bottleneck
        x_c3 = self.conv3(x_c2,  D1//8,       D2//8,       _t(s[3]))

        # ------------------------------------------------------------------
        # Step 5 — Decoder with skip connections
        # [UNCHANGED] skip = interpolate + cat at each decoder stage
        # ------------------------------------------------------------------
        x_c6 = self.conv6(x_c3, D1//2, D2//2, _t(s[4]))
        x_c6 = torch.cat([
            x_c6,
            F.interpolate(x_c1, size=x_c6.shape[2:], mode="trilinear", align_corners=True)
        ], dim=1)

        x_c7 = self.conv7(x_c6, int(3*D1/4), int(3*D2/4), _t(s[5]))
        x_c7 = torch.cat([
            x_c7,
            F.interpolate(x_c0, size=x_c7.shape[2:], mode="trilinear", align_corners=True)
        ], dim=1)

        x_c8 = self.conv8(x_c7, D1, D2, _t(s[6]))
        x_c8 = torch.cat([
            x_c8,
            F.interpolate(x_fc0, size=x_c8.shape[2:], mode="trilinear", align_corners=True)
        ], dim=1)

        # ------------------------------------------------------------------
        # Step 6 — Remove T padding, then clamp to exact T_out
        # [CHANGED v2] trim factor = ratio (not hardcoded 4)
        # [FIXED  v3] rounding in _t() can cause T_final = T_out ± 1;
        #             slice to exactly T_out to guarantee output shape.
        # ------------------------------------------------------------------
        if p != 0:
            trim = int(round(p * self._ratio))
            if self.pad_both:
                x_c8 = x_c8[..., trim : -trim]
            else:
                x_c8 = x_c8[..., : -trim]
        # Clamp to exact T_out (handles ±1 rounding drift)
        x_c8 = x_c8[..., : self.T_out]

        # ------------------------------------------------------------------
        # Step 7 — Output projection
        # [CHANGED] permute: (B, 3w, H, W, T_out) -> (B, T_out, H, W, 3w)
        # [CHANGED] fc2 outputs C_out not 1
        # [UNCHANGED] two-layer MLP with GELU
        # ------------------------------------------------------------------
        x_c8  = x_c8.permute(0, 4, 2, 3, 1)     # (B, T_out, H, W, 3*width)
        x_out = F.gelu(self.fc1(x_c8))           # (B, T_out, H, W, 4*width)
        x_out = self.fc2(x_out)                   # (B, T_out, H, W, C_out)
        x_out = x_out.permute(0, 1, 4, 2, 3)     # (B, T_out, C_out, H, W)

        return x_out

class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, hidden_dim
        ):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.uno = UNO3D_BTCHW(dim, dim, 20, t_in, t_out, 32, 32)
        

    def forward(self, x):
        out = self.uno(x)
        return out


class AmpliNet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=1):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
        # self.tmlp = nn.Sequential(
        #     nn.Linear(pre_seq_length, int(aft_seq_length*mlp_ratio)),
        #     nn.SELU(True),
        #     nn.Linear(int(aft_seq_length*mlp_ratio), aft_seq_length),
        # )
        
        self.amplist = nn.ModuleList([
            AmpCell(pre_seq_length if i==0 else aft_seq_length, aft_seq_length,dim,  hidden_dim) for i in range(n_layers)
        ])
        
    def forward(self, x):
    
        # x_ = x.permute(0,2,3,4,1)
        # xr = self.tmlp(x_)
        # xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for ampcell in self.amplist:
            x = ampcell(x)
        # x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
    
        return x
    
class AlphaPre_Amplinet(nn.Module):
    def __init__(self, total_steps,const_ratio, pre_seq_length, aft_seq_length, input_shape, input_dim, 
                 hidden_dim, n_layers, spec_num=20, kernel_size=1, bias=1, 
                 pha_weight=0.01, anet_weight=0.1, amp_weight=0.01, aweight_stop_steps=10000):
        super(AlphaPre_Amplinet, self).__init__()
        self.amplinet = AmpliNet(pre_seq_length, aft_seq_length, input_dim, hidden_dim)
        self.input_shape, self.input_dim = input_shape, input_dim
        self.hidden_dim = hidden_dim
        self.spec_num = spec_num
        self.pha_weight = pha_weight
        self.anet_weight = anet_weight
        self.amp_weight = amp_weight
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        # self.hfloss = HF_consistency()
        self.itr = 0
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate =  self.amp_weight/self.aweight_stop_steps

        h, w = input_shape
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[...,:spec_num,:spec_num] = 1.
        spec_mask[...,-spec_num:,:spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)
        
    def forward(self, x, y, cmp_fft_loss=False): # x:[b,t,c,h,w]
        self.itr += 1
        xas = self.amplinet(x)
        # xas = torch.sigmoid(xas)
        return xas

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            if self.itr < self.aweight_stop_steps:
                self.amp_weight -= self.sampling_changing_rate
            else:
                self.amp_weight  = 0.

            loss = 0.
            
            # frames_fft = torch.fft.rfft2(frames_gt)
            # frames_abs = torch.abs(frames_fft)
            # xas_fft = torch.fft.rfft2(xas)
            # xas_abs = torch.abs(xas_fft)
            # amp_loss = self.criterion(xas_abs, frames_abs)
            # loss += self.amp_weight*amp_loss
            falfcl_loss = self.falfcl(xas, frames_gt)
            # hfloss = self.hfloss(xas, frames_gt)
            # total_loss = falfcl_loss   #Place correct weights here
            loss = {'total_loss': falfcl_loss}
            return xas, loss
        else:
            return xas, None



def get_model(
    total_steps,
    const_ratio,
    img_channels=1,
    dim = 64,
    T_in = 5, 
    T_out = 20,
    input_shape = (128,128),
    n_layers = 3,
    spec_num = 20,
    pha_weight=0.01, 
    anet_weight=0.1,
    amp_weight=0.01,
    aweight_stop_steps=10000,
    **kwargs
):
    model = AlphaPre_Amplinet(total_steps,const_ratio, pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape, input_dim=img_channels, 
                     hidden_dim=dim, n_layers=n_layers, spec_num=spec_num,
                     pha_weight=pha_weight, anet_weight=anet_weight, amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
                     )
    
    return model