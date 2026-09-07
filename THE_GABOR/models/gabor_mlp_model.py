"""
Controlled Gabor + MLP wavelet model (Experiment 1).

This is a deliberately minimal model used to isolate the Gabor + MLP temporal
modelling mechanism of DAWN-Cast.  Everything that is not needed for that
isolation has been removed.

Architecture
------------
    Input  (B, T_in, C, H, W)
      -> Lifting            (frame-wise, C -> hidden_dim)
      -> spatial DWT        (J-level, per frame)
      -> per subband, independently:
             Gabor  +  MLP  -> simple fusion (concat + 1x1x1 Conv3d)
      -> IDWT
      -> Projection         (frame-wise, hidden_dim -> C)
    Output (B, T_out, C, H, W)

Explicitly NOT present (removed w.r.t. the full DAWN-Cast model):
    SRST / STR / AFNO, spectral refinement, spatial refinement,
    Fourier refinement, the Gabor residual bypass around SRST,
    and the WGTM aggregation logic.

The `GaborLayer`, `_ConvNormAct` and `TransformBlock` definitions are copied
verbatim (up to the `freq_multiplier` default) from
`models/DAWNCast/dawncast.py` so that the Gabor formulation studied here is the
same one used by DAWN-Cast.  The original file is not imported or modified.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse

from utils.utilspp import RandomScheduling   # existing FACL implementation


# ============================================================
# Gabor activation -- copied from models/DAWNCast/dawncast.py
# Only difference: freq_multiplier defaults to 1.0 (no regime prior).
# ============================================================

class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale,
                 alpha=1.0, beta=1.0, freq_multiplier=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)

        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )

        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data = (2 * torch.rand(out_features) - 1) * weight_scale * torch.sqrt(self.gamma)

        # freq ~ Uniform(0, 1)
        self.freq = nn.Parameter(torch.rand(out_features))
        self.freq_multiplier = freq_multiplier

    def squared_distance(self, x):
        """D(x) -- the squared-distance term of the Gaussian envelope."""
        return (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )

    def forward(self, x):
        D = self.squared_distance(x)
        return torch.sin(self.freq_multiplier * self.freq * self.linear(x)) * \
               torch.exp(-0.5 * D * self.gamma[None, :])


# ============================================================
# Lifting / Projection building blocks -- copied from dawncast.py
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
# Gabor + MLP temporal block (one wavelet subband)
# ============================================================

class GaborMLPBlock(nn.Module):
    """
    Gabor + MLP dual-stream temporal modelling for a single wavelet subband,
    followed by a simple fusion (concatenation + 1x1x1 Conv3d).

    Same structure as the DAWN-Cast FAT Block, but only the fused output is
    used -- there is no Gabor residual path in this controlled model.

    Args:
        t_in, t_out     : input / output temporal lengths
        dim             : channel dimension of the subband
        weight_scale    : Gabor weight initialisation scale
        alpha, beta     : Gamma distribution parameters for gamma initialisation
        freq_multiplier : global frequency scaling for the Gabor layer (1.0 here)
        size_factor     : MLP hidden dimension expansion factor
    """
    def __init__(self, t_in, t_out, dim, weight_scale, alpha, beta,
                 freq_multiplier=1.0, size_factor=1.0):
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
        x       : (B, C, H, W, T_in)
        returns : (B, C, T_out, H, W)
        """
        gabor_out = self.gabor(x)                        # (B, C, H, W, T_out)
        mlp_out = self.mlp(x)                            # (B, C, H, W, T_out)

        fused = torch.cat([gabor_out, mlp_out], dim=1)   # (B, 2C, H, W, T_out)
        fused = fused.permute(0, 1, 4, 2, 3)             # (B, 2C, T_out, H, W)
        fused = self.fusion(fused)                       # (B, C,  T_out, H, W)
        return fused


# ============================================================
# Controlled model
# ============================================================

class GaborMLPControlled(nn.Module):
    """
    Lifting -> DWT -> (Gabor + MLP per subband) -> IDWT -> Projection.
    """

    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale=0.1, alpha=1.0, beta=1.0, freq_multiplier=1.0,
                 size_factor=1.0, wave='db6', wavelet_level=2, hf_mode='separate'):
        super().__init__()
        assert wavelet_level in (1, 2, 3, 4), "Levels 1-4 supported"
        assert hf_mode in ('shared', 'separate')

        self.T_in, self.T_out = T_in, T_out
        self.hidden_dim = hidden_dim
        self.wave = wave
        self.level = wavelet_level
        self.hf_mode = hf_mode
        self.freq_multiplier = freq_multiplier

        # ---- Lifting: C -> hidden_dim (frame-wise) ----
        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # ---- Wavelet transform ----
        self.dwt = DWTForward(J=wavelet_level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        # ---- Gabor + MLP block for the LL subband ----
        self.block_ll = GaborMLPBlock(T_in, T_out, hidden_dim,
                                      weight_scale, alpha, beta,
                                      freq_multiplier, size_factor)

        # ---- Gabor + MLP block(s) for the HF subbands ----
        # Identical initialisation prior for every subband: the same
        # freq_multiplier / weight_scale / alpha / beta are used everywhere.
        if hf_mode == 'shared':
            self.block_hf = GaborMLPBlock(T_in, T_out, 3 * hidden_dim,
                                          weight_scale, alpha, beta,
                                          freq_multiplier, size_factor)
        else:
            self.blocks_hf = nn.ModuleList([
                GaborMLPBlock(T_in, T_out, 3 * hidden_dim,
                              weight_scale, alpha, beta,
                              freq_multiplier, size_factor)
                for _ in range(wavelet_level)
            ])

        # ---- Projection: hidden_dim -> C (frame-wise) ----
        self.projection = nn.Sequential(
            TransformBlock(hidden_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1),
        )

    # ------------------------------------------------------------------
    def gabor_layers(self):
        """
        Ordered mapping subband-name -> GaborLayer.
        Names are stable across runs, regimes and seeds.
        """
        layers = OrderedDict()
        layers['LL'] = self.block_ll.gabor
        if self.hf_mode == 'shared':
            layers['HF_shared'] = self.block_hf.gabor
        else:
            for i, blk in enumerate(self.blocks_hf):
                layers[f'HF_level_{i + 1}'] = blk.gabor
        return layers

    # ------------------------------------------------------------------
    def forward(self, x):
        # x: (B, T_in, C, H, W)
        B, T, C, H, W = x.shape

        # ---- Lifting (frame-wise) ----
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.lifting(x)

        # ---- spatial DWT ----
        ll, hf_list = self.dwt(x)
        # ll        : (B*T, C', H_J, W_J)
        # hf_list[i]: (B*T, C', 3, H_i, W_i)

        # ---- Gabor + MLP per subband, independently ----
        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_fused = self.block_ll(ll_t)                        # (B, C', T_out, H_J, W_J)

        hf_fused_list = []
        for i, hf in enumerate(hf_list):
            hf_t = rearrange(hf, '(b t) c n h w -> b (c n) h w t', t=T)
            blk = self.block_hf if self.hf_mode == 'shared' else self.blocks_hf[i]
            hf_fused_list.append(blk(hf_t))                   # (B, 3C', T_out, H_i, W_i)

        # ---- IDWT ----
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon = [rearrange(h, 'b (c n) t h w -> (b t) c n h w', n=3)
                    for h in hf_fused_list]
        recon = self.idwt((ll_recon, hf_recon))
        recon = recon[..., :H, :W]                            # trim padding

        # ---- Projection (frame-wise) ----
        out = self.projection(recon)
        out = rearrange(out, '(b t) c h w -> b t c h w', t=self.T_out)
        return out


# ============================================================
# Forecaster wrapper -- FACL-only training objective
# ============================================================

class GaborMLPForecaster(nn.Module):
    """
    Wraps `GaborMLPControlled` with the existing FACL implementation
    (`utils.utilspp.RandomScheduling`, the same loss DAWN-Cast trains with).

    The training objective is FACL and nothing else:
        total_loss = FACL(prediction, target)
    `predict` returns the *same tensor object* for 'facl_loss' and
    'total_loss', so equality holds exactly (not just numerically).
    """

    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale=0.1, alpha=1.0, beta=1.0, freq_multiplier=1.0,
                 size_factor=1.0, wave='db6', wavelet_level=2, hf_mode='separate',
                 total_steps=50000, const_ratio=0.1):
        super().__init__()
        self.net = GaborMLPControlled(
            T_in=T_in, T_out=T_out, in_dim=in_dim, hidden_dim=hidden_dim,
            weight_scale=weight_scale, alpha=alpha, beta=beta,
            freq_multiplier=freq_multiplier, size_factor=size_factor,
            wave=wave, wavelet_level=wavelet_level, hf_mode=hf_mode,
        )
        self.T_in, self.T_out = T_in, T_out
        self.facl = RandomScheduling(total_steps, 1, const_ratio)

    def gabor_layers(self):
        return self.net.gabor_layers()

    def forward(self, x):
        return self.net(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        pred = self.net(frames_in)
        if not compute_loss:
            return pred, None
        facl_loss = self.facl(pred, frames_gt)
        # total_loss IS facl_loss -- no other term is added anywhere.
        return pred, {'facl_loss': facl_loss, 'total_loss': facl_loss}


def get_model(T_in, T_out, img_channels, dim=64,
              weight_scale=0.1, alpha=1.0, beta=1.0, freq_multiplier=1.0,
              size_factor=1.0, wave='db6', wavelet_level=2, hf_mode='separate',
              total_steps=50000, const_ratio=0.1, **kwargs):
    return GaborMLPForecaster(
        T_in=T_in, T_out=T_out, in_dim=img_channels, hidden_dim=dim,
        weight_scale=weight_scale, alpha=alpha, beta=beta,
        freq_multiplier=freq_multiplier, size_factor=size_factor,
        wave=wave, wavelet_level=wavelet_level, hf_mode=hf_mode,
        total_steps=total_steps, const_ratio=const_ratio,
    )


if __name__ == '__main__':
    for lvl in (1, 2, 3):
        for hf_mode in ('separate', 'shared'):
            m = get_model(T_in=5, T_out=20, img_channels=1, dim=64,
                          wave='db6', wavelet_level=lvl, hf_mode=hf_mode)
            y = m(torch.randn(1, 5, 1, 128, 128))
            n = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
            print(f"J={lvl} {hf_mode:<9} out={tuple(y.shape)} params={n:.2f}M "
                  f"subbands={list(m.gabor_layers().keys())}")
