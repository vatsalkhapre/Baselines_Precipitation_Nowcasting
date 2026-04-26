"""
Wavelet-Gabor LASTOCast Block (Unified)

Supports:
    - J=1 to J=4 with separate Gabor streams per HF level
    - hf_mode: 'shared' (one stream for all HF) or 'separate' (one per level)

Pipeline:
    Input → Lifting → DWT → [Gabor+MLP per band] → Fusion per band → IDWT
          → Spatio-Temporal Conv → + Gabor residual → Projection

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

class AFNO2D(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=1, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

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
        N = H*W
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
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )
        
        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] =  F.relu(
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        
        x = F.softshrink(x, lambd=self.sparsity_threshold)
       
        
        x = torch.view_as_complex(x)
   
        x = x.reshape(B, x.shape[1], x.shape[2], C)
    
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
    
        x = x.type(dtype)
        return x + bias

class SpectralBlock_2D(nn.Module):
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor, k_spatial, groupnorm=True, groups=8):

        super().__init__()

        pad_spatial = (k_spatial - 1) // 2

        self.proj = AFNO2D(dim, num_blocks, sparsity_threshold,
                           hidden_size_factor=hidden_size_factor)

        self.dw_spatial = nn.Conv2d(dim, dim, kernel_size=k_spatial,
                                   padding=pad_spatial, groups=dim, bias=False)

        self.norm = nn.GroupNorm(groups, dim) if groupnorm else nn.BatchNorm2d(dim)

        
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU(),
            nn.Conv2d(dim*2, dim, 1))

        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, H, W, C)
        shortcut = x

        x_ = x.permute(0,3,1,2)
    
        x_spa = self.dw_spatial(x_)

        x_spec = self.proj(x_.permute(0,2,3,1))
        x_spec = x_spec.permute(0,3,1,2)
        
        x_fused = x_spa + x_spec

        x_fused = self.norm(x_fused)

        x_fused = self.act(x_fused)

        x_fused = self.pw(x_fused)

        x_fused = x_fused.permute(0,2,3,1)

        out = x_fused

        return out

class ResneSpectralBlock(nn.Module):
    def __init__(self, dim,num_blocks, sparsity_threshold, hidden_size_factor,  k_spatial, groups = 8): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = SpectralBlock_2D(dim, num_blocks=num_blocks, sparsity_threshold=sparsity_threshold, hidden_size_factor= hidden_size_factor, k_spatial= k_spatial, groups = groups)
        self.block2 = SpectralBlock_2D(dim, num_blocks=num_blocks, sparsity_threshold=sparsity_threshold, hidden_size_factor= hidden_size_factor, k_spatial= k_spatial, groups = groups)
        self.res_conv = nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)
    
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

        return gabor_out, mlp_out, fused


# ============================================================
# Main Wavelet-Gabor Block
# ============================================================

class WaveletGaborBlock(nn.Module):
    """
    LASTOCast block with wavelet-decomposed dual Gabor temporal modeling.

    Args:
        level: DWT decomposition level (1, 2, 3, or 4)
        hf_mode: 'shared' or 'separate'
            - 'shared': single Gabor stream for all HF levels
            - 'separate': one Gabor stream per HF level
    """
    def __init__(self, t_in, t_out, dim, num_blocks,sparsity_threshold, hidden_size_factor,
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
        if hf_mode == 'shared':
            # Single stream shared across all HF levels
            self.stream_hf = BandTemporalStream(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                size_factor,
            )
        else:
            # Separate stream per HF level
            # Interpolate freq_multiplier from high (coarsest) to low (finest)
            self.hf_streams = nn.ModuleList()
            for i in range(level):
                if level == 1:
                    freq_i = freq_multiplier_high
                else:
                    # Level 0 = coarsest (highest freq), level[-1] = finest (mid freq)
                    # Interpolate: coarsest gets freq_high, finest gets midpoint
                    freq_mid = (freq_multiplier_low + freq_multiplier_high) / 2
                    alpha_interp = i / (level - 1)  # 0 for coarsest, 1 for finest
                    freq_i = freq_multiplier_high * (1 - alpha_interp) + freq_mid * alpha_interp

                self.hf_streams.append(BandTemporalStream(
                    t_in, t_out, 3 * dim,
                    weight_scale_high, alpha_high, beta_high, freq_i,
                    size_factor,
                ))
            #     print("freq_i", freq_i)
            
            # print("hf_streams length", len(self.hf_streams))
        # ---- Spatio-Temporal Interaction ----
        # self.spatial_temporal = nn.Sequential(
        #     TransformBlock(dim * t_out, dim * t_out),
        #     TransformBlock(dim * t_out, dim * t_out),
        #     nn.Conv2d(dim * t_out, dim * t_out, kernel_size=3, padding=1),
        # )

        self.conv_spectral = nn.Sequential(ResneSpectralBlock(dim*t_out, num_blocks, sparsity_threshold, hidden_size_factor, k_spatial),
                                     ResneSpectralBlock(dim*t_out, num_blocks, sparsity_threshold, hidden_size_factor, k_spatial),
                                     AFNO2D(dim*t_out, num_blocks, sparsity_threshold, hidden_size_factor= hidden_size_factor))
        self.viz_counter = 0

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape

        # ============================================================
        # 1. DWT decomposition
        # ============================================================
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        # print("x_flat", x_flat.shape)
        ll, hf_list = self.dwt(x_flat)
        # print("hf_list", len(hf_list))

        # for i, ele in enumerate(hf_list):
            # print(i, ele.shape)
        # ll: (B*T, C, H/2^level, W/2^level)
        # hf_list[i]: (B*T, C, 3, H_i, W_i) for i in range(level)
        # hf_list[0] = coarsest, hf_list[-1] = finest

        # ============================================================
        # 2. Temporal processing per band
        # ============================================================

        # --- LL band ---
        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_gabor, ll_mlp, ll_fused = self.stream_ll(ll_t)

        # --- HF bands ---
        hf_gabor_list = []
        hf_fused_list = []
        hf_mlp_list = []

        for i in range(len(hf_list)):
            hf_t = rearrange(hf_list[i], '(b t) c n h w -> b (c n) h w t', t=T)

            if self.hf_mode == 'shared':
                hf_gabor, hf_mlp, hf_fused = self.stream_hf(hf_t)
            else:
                hf_gabor, hf_mlp, hf_fused = self.hf_streams[i](hf_t)

            hf_gabor_list.append(hf_gabor)
            hf_mlp_list.append(hf_mlp)
            hf_fused_list.append(hf_fused)

        # # ===== VISUALIZATION (ONLY FOR DEBUG) =====
        # self.debug_data = {
        #     "ll_gabor": ll_gabor.detach().cpu(),
        #     "hf_gabor": [hf.detach().cpu() for hf in hf_gabor_list]

        # ============================================================
        # 3. IDWT reconstruction (fused path)
        # ============================================================
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = []
        for hf_fused in hf_fused_list:
            hf_recon = rearrange(hf_fused, 'b (c n) t h w -> (b t) c n h w', n=3)
            hf_recon_list.append(hf_recon)

        reconstructed = self.idwt((ll_recon, hf_recon_list))

        # ============================================================
        # 4. IDWT reconstruction (gabor-only residual)
        # ============================================================
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_gabor_flat_list = []
        for hf_gabor in hf_gabor_list:
            hf_gabor_flat = rearrange(hf_gabor, 'b (c n) h w t -> (b t) c n h w', n=3)
            hf_gabor_flat_list.append(hf_gabor_flat)

        gabor_residual = self.idwt((ll_gabor_flat, hf_gabor_flat_list))

        # ============================================================
        # 5. Trim, reshape, S-T Conv, residual
        # ============================================================
        reconstructed = reconstructed[..., :H, :W]
        gabor_residual = gabor_residual[..., :H, :W]

        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)
        gabor_residual = rearrange(gabor_residual, '(b t) c h w -> b t c h w', t=self.t_out)

        # Spatio-Temporal Interaction
        x_st = rearrange(reconstructed, 'b t c h w -> b h w (t c)')
        x_st = self.conv_spectral(x_st)
        x_st = rearrange(x_st, 'b h w (t c) -> b t c h w', t=self.t_out)

        # Gabor residual
        x = x_st + gabor_residual

        return x


# ============================================================
# Full LASTOCast with Wavelet-Gabor
# ============================================================

class WaveletLASTOCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,num_blocks, sparsity_threshold,                  
                 hidden_size_factor, weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
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
            T_in, T_out, hidden_dim, num_blocks, sparsity_threshold, hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,
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
    def __init__(self, T_in, T_out, in_dim, hidden_dim, num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 size_factor, total_steps, const_ratio,
                 k_spatial, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.lastocast = WaveletLASTOCast(
            T_in, T_out, in_dim, hidden_dim,num_blocks, sparsity_threshold, hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,)
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
    afno_blocks, sparsity_threshold, afno_hidden_size_factor, weight_scale_low=1.5, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.5, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    size_factor=1.0,
    total_steps=50000, const_ratio=0.5, k_spatial=3,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128),
    **kwargs
):
    model = WaveletLASTOCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim, num_blocks=afno_blocks, sparsity_threshold=sparsity_threshold, hidden_size_factor=afno_hidden_size_factor,
        weight_scale_low=weight_scale_low, alpha_low=alpha_low,
        beta_low=beta_low, freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high, freq_multiplier_high=freq_multiplier_high,
        size_factor=size_factor, total_steps=total_steps, const_ratio=const_ratio, k_spatial=k_spatial, 
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )
    
    return model



# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    print("=== Wavelet-Gabor LASTOCast Self-Test (J=1 to J=4) ===\n")

    # Spatial sizes at each level for 32x32 input:
    # J=1: LL=16x16, HF=[16x16]
    # J=2: LL=8x8,   HF=[8x8, 16x16]
    # J=3: LL=4x4,   HF=[4x4, 8x8, 16x16]
    # J=4: LL=2x2,   HF=[2x2, 4x4, 8x8, 16x16]

    configs = []
    for wave in ['db6']:
        for level in [2, 3, 4]:
            for hf_mode in ['shared', 'separate']:
                configs.append({
                    'wave': wave, 'level': level,
                    'hf_mode': hf_mode,
                })

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
            x = torch.randn(2, 5, 4, 32, 32)
            out = model(x)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            assert out.shape == (2, 20, 4, 32, 32), f"Shape mismatch: {out.shape}"
            print(f"  [PASS] {tag:<25} | out={tuple(out.shape)} | {params:.2f}M")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {tag:<25} | {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(configs)} configs")