"""
DAWN-Cast NeurIPS-rebuttal ablation building blocks (shared).

Naming follows dawncast.py (STRModule / SRSTBlock / SRSTResBlock / FATBlock /
WGTMBlock / DAWNCast / DAWNCastForecaster). The GaborLayer here is the FROZEN
gamma ("expgabor") variant used in
    alpha_amplinet_latent_FAL_FCL_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_expgabor_nosrst_mse_final.py
(gamma is a registered buffer, bias is gamma-scaled) rather than the learnable
gamma written in dawncast.py.

Ablation ladder (all MSE except #6):
    1. MLP only                               -> MLPOnlyOperator
    2. Wavelet + MLP (separate MLP per level) -> WaveletMLPOperator
    3. Wavelet + Gabor + MLP (no SRST)        -> WGTMBlock(srst_depth=0)
    4. + 1 SRSTResBlock + STRModule           -> WGTMBlock(srst_depth=1)
    5. + 2 SRSTResBlock + STRModule (full)    -> WGTMBlock(srst_depth=2)
    6. Full model + FACL                      -> WGTMBlock(srst_depth=2), FACL loss

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
# Spectral Temporal Refinement (STR) module  -- AFNO-based
# ============================================================

class STRModule(nn.Module):
    """Spectral Temporal Refinement (Fourier-domain), based on AFNO [Guibas et al., ICLR 2022]."""
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
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)

        return x + bias


# ============================================================
# Spectral Refinement Spatio-Temporal (SRST) blocks
# ============================================================

class SRSTBlock(nn.Module):
    """STR (global) + depthwise-conv spatial (local) -> GroupNorm -> SiLU -> channel mixing."""
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 k_spatial, groupnorm=True, groups=8):
        super().__init__()
        pad_spatial = (k_spatial - 1) // 2

        self.str_branch = STRModule(dim, num_blocks, sparsity_threshold,
                                    hidden_size_factor=hidden_size_factor)
        self.spatial_branch = nn.Conv2d(dim, dim, kernel_size=k_spatial,
                                        padding=pad_spatial, groups=dim, bias=False)
        self.norm = nn.GroupNorm(groups, dim) if groupnorm else nn.BatchNorm2d(dim)
        self.channel_mixing = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1))
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, H, W, D)
        x_ = x.permute(0, 3, 1, 2)                        # (B, D, H, W)
        x_spa = self.spatial_branch(x_)
        x_spec = self.str_branch(x_.permute(0, 2, 3, 1))  # (B, H, W, D)
        x_spec = x_spec.permute(0, 3, 1, 2)

        x_fused = x_spa + x_spec
        x_fused = self.norm(x_fused)
        x_fused = self.act(x_fused)
        x_fused = self.channel_mixing(x_fused)
        x_fused = x_fused.permute(0, 2, 3, 1)             # (B, H, W, D)
        return x_fused


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
# Gabor activation -- FROZEN gamma ("expgabor") variant
# ============================================================

class GaborLayer(nn.Module):
    """
    Adaptive Gabor activation with FROZEN gamma (bandwidth fixed at init, not
    learned) and gamma-matched bias scaling. gamma is a registered buffer, so the
    optimizer never sees it, while it still saves/loads via state_dict.
    """
    def __init__(self, in_features, out_features, weight_scale,
                 alpha=1.0, beta=1.0, freq_multiplier=1.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)

        gamma = torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        self.register_buffer('gamma', gamma)      # frozen: buffer, not Parameter

        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data = (2 * torch.rand(out_features) - 1) * weight_scale * torch.sqrt(self.gamma)

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


# ============================================================
# Lifting / Projection utility blocks
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
# Temporal subband blocks
# ============================================================

class FATBlock(nn.Module):
    """Frequency Adaptive Temporal Block: dual-stream Gabor + MLP for one subband."""
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
        # x: (B, C, H, W, T_in)
        gabor_out = self.gabor(x)    # (B, C, H, W, T_out)
        mlp_out = self.mlp(x)        # (B, C, H, W, T_out)

        fused = torch.cat([gabor_out, mlp_out], dim=1)  # (B, 2C, H, W, T_out)
        fused = fused.permute(0, 1, 4, 2, 3)            # (B, 2C, T_out, H, W)
        fused = self.fusion(fused)                        # (B, C, T_out, H, W)
        return gabor_out, mlp_out, fused


class MLPBandBlock(nn.Module):
    """MLP-only temporal block for one subband (ablation 2, no Gabor stream)."""
    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.fusion = nn.Conv3d(dim, dim, kernel_size=1)

    def forward(self, x):
        # x: (B, C, H, W, T_in)
        out = self.mlp(x)                 # (B, C, H, W, T_out)
        out = out.permute(0, 1, 4, 2, 3)  # (B, C, T_out, H, W)
        out = self.fusion(out)            # (B, C, T_out, H, W)
        return out


# ============================================================
# Operators (one per ablation family)
# ============================================================

class MLPOnlyOperator(nn.Module):
    """Ablation 1: temporal MLP on the full-resolution latent (no wavelet)."""
    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.t_out = t_out
        self.mlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        x = rearrange(x, 'b t c h w -> b c h w t')
        x = self.mlp(x)                               # (B, C, H, W, T_out)
        x = rearrange(x, 'b c h w t -> b t c h w')
        return x


class WaveletMLPOperator(nn.Module):
    """Ablation 2: DWT -> separate MLP per subband -> IDWT (no Gabor, no SRST)."""
    def __init__(self, t_in, t_out, dim, size_factor=1.0,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.level = level
        self.hf_mode = hf_mode
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        self.mlp_ll = MLPBandBlock(t_in, t_out, dim, size_factor)
        if hf_mode == 'shared':
            self.mlp_hf = MLPBandBlock(t_in, t_out, 3 * dim, size_factor)
        else:
            self.mlp_hf_streams = nn.ModuleList(
                [MLPBandBlock(t_in, t_out, 3 * dim, size_factor) for _ in range(level)])

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)

        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_fused = self.mlp_ll(ll_t)

        hf_fused_list = []
        for i in range(len(hf_list)):
            hf_t = rearrange(hf_list[i], '(b t) c n h w -> b (c n) h w t', t=T)
            if self.hf_mode == 'shared':
                hf_fused = self.mlp_hf(hf_t)
            else:
                hf_fused = self.mlp_hf_streams[i](hf_t)
            hf_fused_list.append(hf_fused)

        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = [rearrange(hf, 'b (c n) t h w -> (b t) c n h w', n=3)
                         for hf in hf_fused_list]
        reconstructed = self.idwt((ll_recon, hf_recon_list))
        reconstructed = reconstructed[..., :H, :W]
        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)
        return reconstructed


class WGTMBlock(nn.Module):
    """
    Wavelet Guided Temporal Modelling block (ablations 3-6).

    srst_depth controls the SRST stack:
        0 -> no SRST                     (ablation 3)
        1 -> 1 SRSTResBlock + STRModule  (ablation 4)
        2 -> 2 SRSTResBlock + STRModule  (ablations 5 and 6, the full model)
    """
    def __init__(self, t_in, t_out, dim, num_blocks, sparsity_threshold,
                 hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared', srst_depth=2):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim = dim
        self.level = level
        self.hf_mode = hf_mode
        self.srst_depth = srst_depth

        assert level in [1, 2, 3, 4], "Levels 1-4 supported"
        assert hf_mode in ['shared', 'separate']
        assert srst_depth in [0, 1, 2]

        self.wave = wave
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # LL subband FAT block
        self.fat_ll = FATBlock(
            t_in, t_out, dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            size_factor,
        )

        # HF subband FAT block(s)
        if hf_mode == 'shared':
            self.fat_hf = FATBlock(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                size_factor,
            )
        else:
            self.fat_hf_streams = nn.ModuleList()
            for i in range(level):
                if level == 1:
                    freq_i = freq_multiplier_high
                else:
                    freq_mid = (freq_multiplier_low + freq_multiplier_high) / 2
                    alpha_interp = i / (level - 1)
                    freq_i = freq_multiplier_high * (1 - alpha_interp) + freq_mid * alpha_interp
                self.fat_hf_streams.append(FATBlock(
                    t_in, t_out, 3 * dim,
                    weight_scale_high, alpha_high, beta_high, freq_i,
                    size_factor,
                ))

        # SRST stack (optional)
        if srst_depth > 0:
            self.srst = nn.Sequential(
                *[SRSTResBlock(dim * t_out, num_blocks, sparsity_threshold,
                               hidden_size_factor, k_spatial) for _ in range(srst_depth)],
                STRModule(dim * t_out, num_blocks, sparsity_threshold,
                          hidden_size_factor=hidden_size_factor)
            )
        else:
            self.srst = None
        self.viz_counter = 0

    def forward(self, x):
        B, T, C, H, W = x.shape

        # DWT
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)

        # LL FAT
        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_gabor, ll_mlp, ll_fused = self.fat_ll(ll_t)

        # HF FAT
        hf_gabor_list, hf_fused_list = [], []
        for i in range(len(hf_list)):
            hf_t = rearrange(hf_list[i], '(b t) c n h w -> b (c n) h w t', t=T)
            if self.hf_mode == 'shared':
                hf_gabor, hf_mlp, hf_fused = self.fat_hf(hf_t)
            else:
                hf_gabor, hf_mlp, hf_fused = self.fat_hf_streams[i](hf_t)
            hf_gabor_list.append(hf_gabor)
            hf_fused_list.append(hf_fused)

        # IDWT (fused path)
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = [rearrange(hf, 'b (c n) t h w -> (b t) c n h w', n=3)
                         for hf in hf_fused_list]
        reconstructed = self.idwt((ll_recon, hf_recon_list))

        # IDWT (Gabor residual path)
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_gabor_flat_list = [rearrange(hf, 'b (c n) h w t -> (b t) c n h w', n=3)
                              for hf in hf_gabor_list]
        gabor_residual = self.idwt((ll_gabor_flat, hf_gabor_flat_list))

        # Trim + reshape
        reconstructed = reconstructed[..., :H, :W]
        gabor_residual = gabor_residual[..., :H, :W]
        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)
        gabor_residual = rearrange(gabor_residual, '(b t) c h w -> b t c h w', t=self.t_out)

        if self.srst is not None:
            x_srst = rearrange(reconstructed, 'b t c h w -> b h w (t c)')
            x_srst = self.srst(x_srst)
            x_srst = rearrange(x_srst, 'b h w (t c) -> b t c h w', t=self.t_out)
            x = x_srst + gabor_residual
        else:
            x = reconstructed + gabor_residual

        return x


# ============================================================
# DAWN-Cast wrapper (Lifting -> operator -> Projection) + Forecaster
# ============================================================

class DAWNCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim, operator):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.operator = operator
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


class DAWNCastForecaster(nn.Module):
    """DAWN-Cast forecaster; loss_type is 'mse' (ablations 1-5) or 'facl' (ablation 6)."""
    def __init__(self, dawncast, loss_type='mse', total_steps=50000, const_ratio=0.1):
        super().__init__()
        self.dawncast = dawncast
        self.loss_type = loss_type
        if loss_type == 'facl':
            self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        else:
            self.mseloss = nn.MSELoss()
        self.itr = 0

    def forward(self, x, y=None, cmp_fft_loss=False):
        self.itr += 1
        return self.dawncast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            if self.loss_type == 'facl':
                loss_val = self.falfcl(xas, frames_gt)
            else:
                loss_val = self.mseloss(xas, frames_gt)
            return xas, {'total_loss': loss_val}
        return xas, None


# ============================================================
# Factory shared by every ablation get_model
# ============================================================

# ablation id -> SRST depth for the WGTM family (ablations 3-6)
_SRST_DEPTH = {3: 0, 4: 1, 5: 2, 6: 2}


def make_model(
    ablation_id, loss_type,
    afno_blocks, sparsity_threshold, afno_hidden_size_factor,
    weight_scale_low=1.5, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.5, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    size_factor=1.0, total_steps=50000, const_ratio=0.1, k_spatial=3,
    img_channels=1, dim=64, T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128), **kwargs
):
    if ablation_id == 1:
        operator = MLPOnlyOperator(T_in, T_out, dim, size_factor)
    elif ablation_id == 2:
        operator = WaveletMLPOperator(T_in, T_out, dim, size_factor,
                                      wave=wave, level=wavelet_level, hf_mode=hf_mode)
    else:
        operator = WGTMBlock(
            T_in, T_out, dim, afno_blocks, sparsity_threshold, afno_hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, wavelet_level, hf_mode,
            srst_depth=_SRST_DEPTH[ablation_id],
        )

    dawncast = DAWNCast(T_in, T_out, img_channels, dim, operator)
    return DAWNCastForecaster(dawncast, loss_type=loss_type,
                              total_steps=total_steps, const_ratio=const_ratio)
