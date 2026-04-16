"""
Wavelet LASTOCast Block — Experiment 2: MLP-only, 2 sequential MLPs per band (No Gabor)

Pipeline:
    Input → Lifting → DWT → [MLP1 → MLP2 per band] → IDWT
          → Spatio-Temporal Conv → + MLP residual → Projection

MLP structure per band:
    MLP1: t_in  → t_out   (temporal expansion)
    MLP2: t_out → t_out   (temporal refinement)

MLP residual: taken from MLP1 output (before MLP2), mirroring the role
              of the Gabor residual in gabor2.

Changes vs gabor2:
    - GaborLayer removed entirely
    - BandTemporalStream: 2 sequential MLPs, no Gabor, no fusion conv
    - MLP residual (from MLP1) replaces Gabor residual
    - All Gabor hyperparams removed (weight_scale, alpha, beta, freq_multiplier)
"""

import torch
from torch import nn
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from utils.utilspp import RandomScheduling


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
# Temporal stream: 2 sequential MLPs for one band
# ============================================================

class BandTemporalStream(nn.Module):
    """
    Two sequential MLPs for temporal modeling of one frequency band.

    MLP1: t_in  → t_out  (expansion + SELU)
    MLP2: t_out → t_out  (refinement + SELU)

    Residual is taken from MLP1 output.
    """
    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        hidden = int(t_out * size_factor)

        # MLP1: t_in → t_out
        self.mlp1 = nn.Sequential(
            nn.Linear(t_in,  hidden),
            nn.SELU(True),
            nn.Linear(hidden, t_out),
        )
        # MLP2: t_out → t_out
        self.mlp2 = nn.Sequential(
            nn.Linear(t_out, hidden),
            nn.SELU(True),
            nn.Linear(hidden, t_out),
        )

    def forward(self, x):
        """
        x:            (B, C, H, W, T_in)
        returns:
            mlp1_out  (B, C, H, W, T_out)   ← for residual path
            fused     (B, C, T_out, H, W)    ← for IDWT path (after MLP2)
        """
        mlp1_out = self.mlp1(x)              # (B, C, H, W, T_out)
        mlp2_out = self.mlp2(mlp1_out)       # (B, C, H, W, T_out)
        fused    = mlp2_out.permute(0, 1, 4, 2, 3)  # (B, C, T_out, H, W)
        
        return mlp1_out, fused


# ============================================================
# Main Wavelet Block (MLP-only, 2 MLPs per band)
# ============================================================

class WaveletMLPBlock(nn.Module):
    def __init__(self, t_in, t_out, dim,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim = dim
        self.level = level
        self.hf_mode = hf_mode

        assert level in [1, 2, 3, 4]
        assert hf_mode in ['shared', 'separate']

        self.dwt  = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # LL stream
        self.stream_ll = BandTemporalStream(t_in, t_out, dim, size_factor)

        # HF streams
        if hf_mode == 'shared':
            self.stream_hf = BandTemporalStream(t_in, t_out, 3 * dim, size_factor)
        else:
            self.hf_streams = nn.ModuleList([
                BandTemporalStream(t_in, t_out, 3 * dim, size_factor)
                for _ in range(level)
            ])

        # Spatio-Temporal Interaction
        self.spatial_temporal = nn.Sequential(
            TransformBlock(dim * t_out, dim * t_out),
            TransformBlock(dim * t_out, dim * t_out),
            nn.Conv2d(dim * t_out, dim * t_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        # 1. DWT
        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)

        # 2. Temporal processing
        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_mlp1, ll_fused = self.stream_ll(ll_t)   # ll_mlp1 = MLP1 output (residual)

        hf_mlp1_list  = []
        hf_fused_list = []
        for i, hf in enumerate(hf_list):
            hf_t = rearrange(hf, '(b t) c n h w -> b (c n) h w t', t=T)
            if self.hf_mode == 'shared':
                hf_mlp1, hf_fused = self.stream_hf(hf_t)
            else:
                hf_mlp1, hf_fused = self.hf_streams[i](hf_t)
            hf_mlp1_list.append(hf_mlp1)
            hf_fused_list.append(hf_fused)

        # 3. IDWT — fused path (MLP2 output)
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = [
            rearrange(hf, 'b (c n) t h w -> (b t) c n h w', n=3)
            for hf in hf_fused_list
        ]
        reconstructed = self.idwt((ll_recon, hf_recon_list))

        # 4. IDWT — MLP1 residual path
        ll_mlp1_flat = rearrange(ll_mlp1, 'b c h w t -> (b t) c h w')
        hf_mlp1_flat_list = [
            rearrange(hf, 'b (c n) h w t -> (b t) c n h w', n=3)
            for hf in hf_mlp1_list
        ]
        mlp_residual = self.idwt((ll_mlp1_flat, hf_mlp1_flat_list))

        # 5. Trim + S-T Conv + residual
        reconstructed = reconstructed[..., :H, :W]
        mlp_residual  = mlp_residual[..., :H, :W]

        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)
        mlp_residual  = rearrange(mlp_residual,  '(b t) c h w -> b t c h w', t=self.t_out)

        x_st = rearrange(reconstructed, 'b t c h w -> b (t c) h w')
        x_st = self.spatial_temporal(x_st)
        x_st = rearrange(x_st, 'b (t c) h w -> b t c h w', t=self.t_out)

        return x_st + mlp_residual


# ============================================================
# Full model
# ============================================================

class WaveletLASTOCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 size_factor=1.0, wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out

        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.operator = WaveletMLPBlock(T_in, T_out, hidden_dim, size_factor, wave, level, hf_mode)
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
                 size_factor, total_steps, const_ratio,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.lastocast = WaveletLASTOCast(T_in, T_out, in_dim, hidden_dim,
                                          size_factor, wave, level, hf_mode)
        self.T_in  = T_in
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
            return xas, {'total_loss': falfcl_loss}
        return xas, None


# ============================================================
# Model Factory
# ============================================================

def get_model(
    size_factor=1.0,
    total_steps=50000, const_ratio=0.5,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128),
    **kwargs
):
    return WaveletLASTOCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )


# ============================================================
# Self-test
# ============================================================
if __name__ == '__main__':
    print("=== Exp 2: Wavelet MLP-only (2 sequential MLPs per band) ===\n")
    configs = [
        {'wave': 'db6', 'level': l, 'hf_mode': m}
        for l in [2, 3, 4] for m in ['shared', 'separate']
    ]
    passed = failed = 0
    for cfg in configs:
        tag = f"{cfg['wave']}_J{cfg['level']}_{cfg['hf_mode']}"
        try:
            model = get_model(img_channels=4, dim=64, T_in=5, T_out=20, **cfg, wavelet_level=cfg['level'])
            x   = torch.randn(2, 5, 4, 32, 32)
            out = model(x)
            params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            assert out.shape == (2, 20, 4, 32, 32)
            print(f"  [PASS] {tag:<25} | out={tuple(out.shape)} | {params:.2f}M")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {tag:<25} | {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(configs)} configs")
