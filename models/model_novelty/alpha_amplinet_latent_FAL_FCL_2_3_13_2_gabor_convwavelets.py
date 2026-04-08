"""
Wavelet LASTOCast v2

New pipeline:
    Input → Lifting → Gabor+MLP Temporal Modeling (full resolution) → Fusion
          → DWT → Spatio-Temporal Conv (per band) → IDWT
          → Residual connection → Projection

Supports:
    - J=1: S-T Conv for LL + S-T Conv for HF
    - J=2: S-T Conv for LL + shared S-T Conv for both HF levels
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
        self.param = nn.Parameter(torch.rand(out_features))
        
        # NOW TRAINABLE - initialized differently for low vs high freq
        self.freq_multiplier = freq_multiplier
        
    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.freq_multiplier * self.param * self.linear(x)) * \
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
# Core Operator Block
# ============================================================

class WaveletGaborBlockV2(nn.Module):
    """
    LASTOCast block with wavelet-decomposed spatio-temporal interaction.

    Pipeline:
        1. Spectral Temporal Modeling (Gabor + MLP on full resolution)
        2. Spectro-Temporal Fusion
        3. DWT decomposition
        4. Spatio-Temporal Conv per wavelet band
        5. IDWT reconstruction
        6. Residual connection (gabor / mlp / none)

    Args:
        level: DWT decomposition level (1 or 2)
        wave: wavelet type ('haar', 'db2', 'db3', 'coif1')
        residual_mode: 'gabor', 'mlp', or 'none'
    """
    def __init__(self, t_in, t_out, dim,
                 weight_scale, alpha, beta, freq_multiplier,
                 size_factor=1.0, wave='haar', level=1, residual_mode='gabor'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim = dim
        self.level = level
        self.wave = wave
        self.residual_mode = residual_mode

        assert level in [1, 2], "Only level 1 and 2 supported"
        assert residual_mode in ['gabor', 'mlp', 'none']

        # ---- Spectral Temporal Modeling (full resolution) ----
        self.gabor = GaborLayer(t_in, t_out, weight_scale, alpha, beta, freq_multiplier)
        self.mlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )

        # ---- Spectro-Temporal Fusion ----
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

        # ---- Wavelet transform ----
        self.dwt = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # ---- Spatio-Temporal Interaction per band ----
        # LL band: operates at reduced resolution
        if level == 1:
            ll_spatial = dim * t_out  # after rearrange (t c) at half resolution
        else:
            ll_spatial = dim * t_out  # at quarter resolution

        self.st_conv_ll = nn.Sequential(
            TransformBlock(ll_spatial, ll_spatial),
            TransformBlock(ll_spatial, ll_spatial),
            nn.Conv2d(ll_spatial, ll_spatial, kernel_size=3, padding=1),
        )

        # HF band: shared across all HF levels for J=2
        hf_spatial = 3 * dim * t_out  # 3 sub-bands flattened with time
        self.st_conv_hf = nn.Sequential(
            TransformBlock(hf_spatial, hf_spatial),
            TransformBlock(hf_spatial, hf_spatial),
            nn.Conv2d(hf_spatial, hf_spatial, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape

        # ============================================================
        # 1. Spectral Temporal Modeling (full resolution)
        # ============================================================
        x_perm = x.permute(0, 2, 3, 4, 1)              # (B, C, H, W, T_in)
        gabor_out = self.gabor(x_perm)                    # (B, C, H, W, T_out)
        mlp_out = self.mlp(x_perm)                        # (B, C, H, W, T_out)

        # ============================================================
        # 2. Spectro-Temporal Fusion
        # ============================================================
        fused = torch.cat([gabor_out, mlp_out], dim=1)    # (B, 2C, H, W, T_out)
        fused = fused.permute(0, 1, 4, 2, 3)              # (B, 2C, T_out, H, W)
        fused = self.fusion(fused)                          # (B, C, T_out, H, W)
        fused = fused.permute(0, 2, 1, 3, 4)              # (B, T_out, C, H, W)

        # ============================================================
        # 3. DWT decomposition
        # ============================================================
        fused_flat = rearrange(fused, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(fused_flat)
        # J=1: ll (B*T_out, C, H/2, W/2), hf_list[0] (B*T_out, C, 3, H/2, W/2)
        # J=2: ll (B*T_out, C, H/4, W/4), hf_list[0] (B*T_out, C, 3, H/4, W/4),
        #       hf_list[1] (B*T_out, C, 3, H/2, W/2)

        # ============================================================
        # 4. Spatio-Temporal Conv per band
        # ============================================================

        # --- LL band ---
        ll_h, ll_w = ll.shape[-2], ll.shape[-1]
        ll = rearrange(ll, '(b t) c h w -> b (t c) h w', t=self.t_out)
        ll = self.st_conv_ll(ll)
        ll = rearrange(ll, 'b (t c) h w -> (b t) c h w', t=self.t_out)

        # --- HF bands ---
        processed_hf = []
        for i in range(len(hf_list)):
            hf = hf_list[i]  # (B*T_out, C, 3, H', W')
            hf_h, hf_w = hf.shape[-2], hf.shape[-1]
            # Flatten sub-bands into channels: (B*T_out, 3C, H', W')
            hf = rearrange(hf, 'bt c n h w -> bt (c n) h w')
            # Merge time for S-T Conv: (B, T_out*3C, H', W')
            hf = rearrange(hf, '(b t) c h w -> b (t c) h w', t=self.t_out)
            hf = self.st_conv_hf(hf)  # shared weights for all HF levels
            # Restore shape
            hf = rearrange(hf, 'b (t c) h w -> (b t) c h w', t=self.t_out)
            # Back to DWT format: (B*T_out, C, 3, H', W')
            hf = rearrange(hf, 'bt (c n) h w -> bt c n h w', n=3)
            processed_hf.append(hf)

        # ============================================================
        # 5. IDWT reconstruction
        # ============================================================
        reconstructed = self.idwt((ll, processed_hf))
        # Trim to original spatial size (in case of padding)
        reconstructed = reconstructed[..., :H, :W]
        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)

        # ============================================================
        # 6. Residual connection
        # ============================================================
        if self.residual_mode == 'gabor':
            residual = gabor_out.permute(0, 4, 1, 2, 3)   # (B, T_out, C, H, W)
            output = reconstructed + residual
        elif self.residual_mode == 'mlp':
            residual = mlp_out.permute(0, 4, 1, 2, 3)     # (B, T_out, C, H, W)
            output = reconstructed + residual
        else:
            output = reconstructed

        return output


# ============================================================
# Full Model
# ============================================================

class WaveletLASTOCastV2(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale, alpha, beta, freq_multiplier,
                 size_factor=1.0, wave='haar', level=1, residual_mode='gabor'):
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
        self.operator = WaveletGaborBlockV2(
            T_in, T_out, hidden_dim,
            weight_scale, alpha, beta, freq_multiplier,
            size_factor, wave, level, residual_mode,
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


class WaveletLASTOCastV2Forecaster(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale, alpha, beta, freq_multiplier,
                 size_factor, total_steps, const_ratio,
                 wave='haar', level=1, residual_mode='gabor'):
        super().__init__()
        self.lastocast = WaveletLASTOCastV2(
            T_in, T_out, in_dim, hidden_dim,
            weight_scale, alpha, beta, freq_multiplier,
            size_factor, wave, level, residual_mode,
        )
        self.T_in = T_in
        self.T_out = T_out
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)

    def forward(self, x, y=None, cmp_fft_loss=False):
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
    weight_scale=1.5, alpha=1.0, beta=1.0, freq_multiplier=1.0,
    size_factor=1.0,
    total_steps=50000, const_ratio=0.5,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, residual_mode='gabor',
    **kwargs
):
    model = WaveletLASTOCastV2Forecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        weight_scale=weight_scale, alpha=alpha,
        beta=beta, freq_multiplier=freq_multiplier,
        size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        wave=wave, level=wavelet_level, residual_mode=residual_mode,
    )
    return model


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    print("=== Wavelet LASTOCast V2 Self-Test ===\n")

    configs = []
    for wave in ['haar', 'db2', 'db3', 'coif1']:
        for level in [1, 2]:
            for res_mode in ['gabor', 'mlp', 'none']:
                configs.append({'wave': wave, 'level': level, 'residual_mode': res_mode})

    passed = 0
    failed = 0
    for cfg in configs:
        tag = f"{cfg['wave']}_J{cfg['level']}_{cfg['residual_mode']}"
        try:
            model = get_model(
                img_channels=4, dim=64,
                T_in=5, T_out=20,
                wave=cfg['wave'], level=cfg['level'],
                residual_mode=cfg['residual_mode'],
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