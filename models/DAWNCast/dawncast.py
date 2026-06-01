"""
DAWN-Cast: Latent Dynamical Adaptive Wavelet Network for Precipitation Nowcasting

Architecture:
    Input → Lifting → DWT → [FAT Block per subband] → IDWT (fused path)
                                                      → IDWT (Gabor residual path)
          → SRST Block → + Gabor residual → Projection

Components:
    - FATBlock        : Frequency Adaptive Temporal Block (Gabor + MLP dual-stream per subband)
    - WGTMBlock       : Wavelet Guided Temporal Modelling Block
    - SRSTBlock       : Spectral Refinement Spatio-Temporal Block
    - SRSTResBlock    : Stack of two SRSTBlocks with residual connection
    - STRModule       : Spectral Temporal Refinement module (Fourier-domain, based on AFNO [1])
    - DAWNCast        : Full DAWN-Cast model
    - DAWNCastForecaster : DAWN-Cast with loss scheduling

[1] Guibas et al., "Adaptive Fourier Neural Operators: Efficient Token Mixers
    for Transformers", ICLR 2022.

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
import matplotlib.pyplot as plt
import os
import time


# ============================================================
# Spectral Temporal Refinement Module
# Based on AFNO (Adaptive Fourier Neural Operator) [Guibas et al., ICLR 2022]
# Used inside SRSTBlock as the global temporal refinement branch
# ============================================================

class STRModule(nn.Module):
    """
    Spectral Temporal Refinement (STR) module.
    Performs Fourier-domain temporal processing for global structure refinement.
    Based on AFNO [Guibas et al., ICLR 2022].

    Args:
        hidden_size             : merged temporal-channel dimension D = T_out * C
        num_blocks              : number of groups partitioning D (higher => fewer parameters)
        sparsity_threshold      : lambda for soft-shrinkage regularization
        hard_thresholding_fraction : fraction of frequency modes retained
        hidden_size_factor      : expansion factor rho_h for hidden dimension d_h = d * rho_h
    """
    def __init__(self, hidden_size, num_blocks=1, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, \
            f"hidden_size {hidden_size} should be divisible by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
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
        x = x.reshape(B, H, W, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = N // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        # Layer 1: complex-valued Fourier-domain transformation (expand to d_h)
        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) +
            self.b1[0]
        )
        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) +
            self.b1[1]
        )

        # Layer 2: complex-valued Fourier-domain transformation (project back to d)
        o2_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) +
            self.b2[0]
        )
        o2_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) +
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)

        # Sparsity regularization via soft-shrinkage
        x = F.softshrink(x, lambd=self.sparsity_threshold)

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias  # residual connection


# ============================================================
# Spectral Refinement Spatio-Temporal (SRST) Block
# Two parallel branches: STR (global) + Spatial (local) → GroupNorm → Channel Mixing
# ============================================================

class SRSTBlock(nn.Module):
    """
    Spectral Refinement Spatio-Temporal (SRST) Block.

    Refines the aggregated latent tensor through two parallel branches:
      - STR branch    : Spectral Temporal Refinement (global, Fourier-domain)
      - Spatial branch: Depthwise convolution for local fine-detail refinement

    Outputs are summed → GroupNorm → SiLU → Channel Mixing (1x1 conv).
    """
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 k_spatial, groupnorm=True, groups=8):
        super().__init__()

        pad_spatial = (k_spatial - 1) // 2

        # STR branch: global Fourier-domain temporal refinement
        self.str_branch = STRModule(dim, num_blocks, sparsity_threshold,
                                    hidden_size_factor=hidden_size_factor)

        # Spatial branch: local fine-detail refinement via depthwise convolution
        self.spatial_branch = nn.Conv2d(dim, dim, kernel_size=k_spatial,
                                        padding=pad_spatial, groups=dim, bias=False)

        self.norm = nn.GroupNorm(groups, dim) if groupnorm else nn.BatchNorm2d(dim)

        # Channel mixing: couples STR and spatial features across temporal-channel dim
        self.channel_mixing = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1))

        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, H, W, D) where D = T_out * C
        shortcut = x

        x_ = x.permute(0, 3, 1, 2)          # (B, D, H, W)

        # Spatial branch: local fine-scale spatial refinement
        x_spa = self.spatial_branch(x_)       # (B, D, H, W)

        # STR branch: global spectral temporal refinement
        x_spec = self.str_branch(x_.permute(0, 2, 3, 1))  # (B, H, W, D)
        x_spec = x_spec.permute(0, 3, 1, 2)               # (B, D, H, W)

        # Fuse branches, normalize, activate, and apply channel mixing
        x_fused = x_spa + x_spec
        x_fused = self.norm(x_fused)
        x_fused = self.act(x_fused)
        x_fused = self.channel_mixing(x_fused)

        x_fused = x_fused.permute(0, 2, 3, 1)  # (B, H, W, D)

        return x_fused


# ============================================================
# SRST Residual Block: two SRSTBlocks with skip connection
# ============================================================

class SRSTResBlock(nn.Module):
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 k_spatial, groups=8):
        super().__init__()
        self.srst_block1 = SRSTBlock(dim, num_blocks=num_blocks,
                                     sparsity_threshold=sparsity_threshold,
                                     hidden_size_factor=hidden_size_factor,
                                     k_spatial=k_spatial, groups=groups)
        self.srst_block2 = SRSTBlock(dim, num_blocks=num_blocks,
                                     sparsity_threshold=sparsity_threshold,
                                     hidden_size_factor=hidden_size_factor,
                                     k_spatial=k_spatial, groups=groups)
        self.res_conv = nn.Identity()

    def forward(self, x):
        h = self.srst_block1(x)
        h = self.srst_block2(h)
        return h + self.res_conv(x)


# ============================================================
# Gabor Activation (used inside FATBlock)
# ============================================================

class GaborLayer(nn.Module):
    """
    Adaptive Gabor activation for subband-specific climatic inductive bias.

    Learnable parameters:
        mu    : Gaussian envelope center (localization in feature space)
        gamma : bandwidth (initialized from Gamma(alpha, beta) prior)
        freq  : per-neuron base frequency
        freq_multiplier : global frequency scaling lambda

    Behavioral regimes (see paper Section 3):
        - Small lambda*freq, small gamma => near-linear (slowly evolving subbands)
        - Large lambda*freq, large gamma => oscillatory + localized (turbulent subbands)
    """
    def __init__(self, in_features, out_features, weight_scale,
                 alpha=1.0, beta=1.0, freq_multiplier=1.5):
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
        # Distance term D(x) = ||x - mu||^2
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        # Gabor activation: sin(lambda * f * W^T x) * exp(-0.5 * gamma * D)
        return torch.sin(self.freq_multiplier * self.freq * self.linear(x)) * \
               torch.exp(-0.5 * D * self.gamma[None, :])


# ============================================================
# Utility blocks for Lifting and Projection
# ============================================================

class _ConvNormAct(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size,
                              padding=kernel_size // 2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class TransformBlock(nn.Module):
    """Residual Conv block used in Lifting and Projection."""
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.block1 = _ConvNormAct(dim, dim_out, groups=groups,
                                   kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = _ConvNormAct(dim_out, dim_out, groups=groups,
                                   kernel_size=kernel_size, padding_mode=padding_mode)
        self.skip = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


# ============================================================
# FAT Block: Frequency Adaptive Temporal Block
# Dual-stream (Gabor + MLP) temporal modeling for a single wavelet subband
# ============================================================

class FATBlock(nn.Module):
    """
    Frequency Adaptive Temporal (FAT) Block.

    Applied independently to each wavelet subband. Combines:
      - Gabor stream : adaptive Gabor activation providing subband-specific
                       climatic inductive bias (see GaborLayer)
      - MLP stream   : stable nonlinear temporal transformation

    The two streams are fused via concatenation + 1x1x1 Conv3d.

    Args:
        t_in, t_out     : input/output temporal lengths
        dim             : channel dimension of the subband
        weight_scale    : Gabor weight initialization scale
        alpha, beta     : Gamma distribution parameters for gamma initialization
        freq_multiplier : global frequency scaling lambda for Gabor
        size_factor     : MLP hidden dimension expansion factor
    """
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
        x         : (B, C, H, W, T_in)
        returns   : gabor_out  (B, C, H, W, T_out)  -- Gabor stream output (for residual path)
                    mlp_out    (B, C, H, W, T_out)  -- MLP stream output
                    fused_out  (B, C, T_out, H, W)  -- fused output (for WGTM aggregation)
        """
        gabor_out = self.gabor(x)    # (B, C, H, W, T_out)
        mlp_out   = self.mlp(x)      # (B, C, H, W, T_out)

        # Concatenate along channel dim, permute for Conv3d, fuse 2C → C
        fused = torch.cat([gabor_out, mlp_out], dim=1)  # (B, 2C, H, W, T_out)
        fused = fused.permute(0, 1, 4, 2, 3)            # (B, 2C, T_out, H, W)
        fused = self.fusion(fused)                        # (B, C, T_out, H, W)

        return gabor_out, mlp_out, fused


# ============================================================
# WGTM Block: Wavelet Guided Temporal Modelling Block
# DWT → FAT Block per subband → IDWT (fused) + IDWT (Gabor residual) → SRST
# ============================================================

class WGTMBlock(nn.Module):
    """
    Wavelet Guided Temporal Modelling (WGTM) Block.

    Pipeline:
        1. J-level 2D DWT → LL subband + HF subbands per level
        2. FATBlock applied independently to each subband
        3. IDWT of fused FAT outputs  → input to SRST Block
        4. IDWT of Gabor-only outputs → Gabor residual (added after SRST)
        5. SRST Block refines the fused latent
        6. Add Gabor residual → final WGTM output

    Args:
        level   : DWT decomposition level J (1-4)
        hf_mode : 'shared' (one FATBlock for all HF levels) or
                  'separate' (one FATBlock per HF level)
    """
    def __init__(self, t_in, t_out, dim, num_blocks, sparsity_threshold,
                 hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim = dim
        self.level = level
        self.hf_mode = hf_mode

        assert level in [1, 2, 3, 4], "Levels 1-4 supported"
        assert hf_mode in ['shared', 'separate']

        # ---- Wavelet transform (J-level DWT / IDWT) ----
        self.wave = wave
        self.dwt  = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # ---- FAT Block for LL subband (low-frequency, large-scale convective structure) ----
        self.fat_ll = FATBlock(
            t_in, t_out, dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            size_factor,
        )

        # ---- FAT Block(s) for HF subbands (high-frequency, turbulent fine-scale variability) ----
        if hf_mode == 'shared':
            # Single FAT Block shared across all HF levels
            self.fat_hf = FATBlock(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                size_factor,
            )
        else:
            # Separate FAT Block per HF level (frequency interpolated across levels)
            self.fat_hf_streams = nn.ModuleList()
            for i in range(level):
                if level == 1:
                    freq_i = freq_multiplier_high
                else:
                    # Interpolate: coarsest level gets freq_high, finest gets midpoint
                    freq_mid = (freq_multiplier_low + freq_multiplier_high) / 2
                    alpha_interp = i / (level - 1)
                    freq_i = freq_multiplier_high * (1 - alpha_interp) + freq_mid * alpha_interp

                self.fat_hf_streams.append(FATBlock(
                    t_in, t_out, 3 * dim,
                    weight_scale_high, alpha_high, beta_high, freq_i,
                    size_factor,
                ))

        # ---- SRST Block stack: refines the IDWT-aggregated fused outputs ----
        self.srst = nn.Sequential(
            SRSTResBlock(dim * t_out, num_blocks, sparsity_threshold,
                         hidden_size_factor, k_spatial),
            SRSTResBlock(dim * t_out, num_blocks, sparsity_threshold,
                         hidden_size_factor, k_spatial),
            STRModule(dim * t_out, num_blocks, sparsity_threshold,
                      hidden_size_factor=hidden_size_factor)
        )
        self.viz_counter = 0

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape

        # ============================================================
        # 1. J-level 2D DWT decomposition
        # ============================================================
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)
        # ll        : (B*T, C, H/2^J, W/2^J)         — LL subband
        # hf_list[i]: (B*T, C, 3, H_i, W_i)          — HF subbands per level

        # ============================================================
        # 2. FAT Block temporal processing per subband
        # ============================================================

        # --- LL subband → FAT Block (Low) ---
        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_gabor, ll_mlp, ll_fused = self.fat_ll(ll_t)

        # --- HF subbands → FAT Block (High) per level ---
        hf_gabor_list  = []
        hf_fused_list  = []
        hf_mlp_list    = []

        for i in range(len(hf_list)):
            hf_t = rearrange(hf_list[i], '(b t) c n h w -> b (c n) h w t', t=T)

            if self.hf_mode == 'shared':
                hf_gabor, hf_mlp, hf_fused = self.fat_hf(hf_t)
            else:
                hf_gabor, hf_mlp, hf_fused = self.fat_hf_streams[i](hf_t)

            hf_gabor_list.append(hf_gabor)
            hf_mlp_list.append(hf_mlp)
            hf_fused_list.append(hf_fused)

        # ============================================================
        # 3. IDWT reconstruction — fused path (input to SRST Block)
        # ============================================================
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = []
        for hf_fused in hf_fused_list:
            hf_recon = rearrange(hf_fused, 'b (c n) t h w -> (b t) c n h w', n=3)
            hf_recon_list.append(hf_recon)

        reconstructed = self.idwt((ll_recon, hf_recon_list))

        # ============================================================
        # 4. IDWT reconstruction — Gabor residual path (bypasses SRST)
        # ============================================================
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_gabor_flat_list = []
        for hf_gabor in hf_gabor_list:
            hf_gabor_flat = rearrange(hf_gabor, 'b (c n) h w t -> (b t) c n h w', n=3)
            hf_gabor_flat_list.append(hf_gabor_flat)

        gabor_residual = self.idwt((ll_gabor_flat, hf_gabor_flat_list))

        # ============================================================
        # 5. Trim, reshape, apply SRST Block, add Gabor residual
        # ============================================================
        reconstructed  = reconstructed[..., :H, :W]
        gabor_residual = gabor_residual[..., :H, :W]

        reconstructed  = rearrange(reconstructed,  '(b t) c h w -> b t c h w', t=self.t_out)
        gabor_residual = rearrange(gabor_residual,  '(b t) c h w -> b t c h w', t=self.t_out)

        # SRST Block: refines fused latent in merged temporal-channel space
        x_srst = rearrange(reconstructed, 'b t c h w -> b h w (t c)')
        x_srst = self.srst(x_srst)
        x_srst = rearrange(x_srst, 'b h w (t c) -> b t c h w', t=self.t_out)

        # Final output: SRST-refined + Gabor residual (Eq. Z_hat = SRST(Z_fused) + R_Gabor)
        x = x_srst + gabor_residual

        return x


# ============================================================
# DAWN-Cast: Full model (Lifting → WGTM → Projection)
# ============================================================

class DAWNCast(nn.Module):
    """
    DAWN-Cast: Latent Dynamical Adaptive Wavelet Network.

    Full pipeline:
        Input (B, T_in, C, H, W)
        → Lifting (c → c')
        → WGTMBlock (wavelet decomp + FAT Blocks + SRST Block)
        → Projection (c' → c)
        → Output (B, T_out, C, H, W)
    """
    def __init__(self, T_in, T_out, in_dim, hidden_dim, num_blocks,
                 sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out

        # Lifting: c → c' (expand latent channels)
        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # WGTM Block: wavelet-guided temporal modelling
        self.wgtm = WGTMBlock(
            T_in, T_out, hidden_dim, num_blocks, sparsity_threshold,
            hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,
        )

        # Projection: c' → c (reduce back to latent channels)
        self.projection = nn.Sequential(
            TransformBlock(hidden_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1),
        )

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.lifting(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_in)

        x = self.wgtm(x)

        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.projection(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_out)
        return x


# ============================================================
# DAWN-Cast Forecaster (with FACL loss scheduling)
# ============================================================

class DAWNCastForecaster(nn.Module):
    """
    DAWN-Cast Forecaster wrapper with Fourier Amplitude and Correlation Loss (FACL)
    scheduling via RandomScheduling.
    """
    def __init__(self, T_in, T_out, in_dim, hidden_dim, num_blocks,
                 sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor, total_steps, const_ratio,
                 k_spatial, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.dawncast = DAWNCast(
            T_in, T_out, in_dim, hidden_dim, num_blocks, sparsity_threshold,
            hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,
        )
        self.T_in  = T_in
        self.T_out = T_out
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        self.itr = 0

    def forward(self, x, y=None, cmp_fft_loss=False):
        self.itr += 1
        return self.dawncast(x)

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
    afno_blocks, sparsity_threshold, afno_hidden_size_factor,
    weight_scale_low=1.5, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.5, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    size_factor=1.0,
    total_steps=50000, const_ratio=0.5, k_spatial=3,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128),
    **kwargs
):
    model = DAWNCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        num_blocks=afno_blocks,
        sparsity_threshold=sparsity_threshold,
        hidden_size_factor=afno_hidden_size_factor,
        weight_scale_low=weight_scale_low,   alpha_low=alpha_low,
        beta_low=beta_low,                   freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high,                 freq_multiplier_high=freq_multiplier_high,
        size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        k_spatial=k_spatial,
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )
    return model


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    print("=== DAWN-Cast Self-Test (J=1 to J=4) ===\n")

    configs = []
    for wave in ['db6']:
        for level in [2, 3, 4]:
            for hf_mode in ['shared', 'separate']:
                configs.append({'wave': wave, 'level': level, 'hf_mode': hf_mode})

    passed = 0
    failed = 0
    for cfg in configs:
        tag = f"{cfg['wave']}_J{cfg['level']}_{cfg['hf_mode']}"
        try:
            model = get_model(
                img_channels=4, dim=64,
                T_in=5, T_out=20,
                wave=cfg['wave'], wavelet_level=cfg['level'],
                hf_mode=cfg['hf_mode'],
            )
            x   = torch.randn(2, 5, 4, 32, 32)
            out = model(x)
            params = sum(p.numel() for p in model.parameters()
                         if p.requires_grad) / 1e6
            assert out.shape == (2, 20, 4, 32, 32), f"Shape mismatch: {out.shape}"
            print(f"  [PASS] {tag:<25} | out={tuple(out.shape)} | {params:.2f}M")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {tag:<25} | {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(configs)} configs")