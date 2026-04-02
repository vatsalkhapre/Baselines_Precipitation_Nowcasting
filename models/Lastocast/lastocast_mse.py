"""
LASTOCast: Latent Spectro-Temporal Operator for Precipitation Nowcasting

Architecture Overview:
    Input (B, T_in, C, H, W)
        → Lifting (latent-to-dynamics extractor)
        → Spectral Temporal Modeling (Gabor + MLP dual-stream)
        → Spectro-Temporal Fusion (learned per-pixel gating)
        → Spatio-Temporal Interaction (joint spatial-temporal operator)
        → Residual connection from Gabor stream
        → Projection (dynamics-to-latent projector)
    Output (B, T_out, C, H, W)
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from utils.utilspp import RandomScheduling


# ============================================================
# Building Blocks
# ============================================================

class ConvBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size,
                              padding=kernel_size // 2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class ConvTransformBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.block1 = ConvBlock(dim, dim_out, groups=groups,
                            kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = ConvBlock(dim_out, dim_out, groups=groups,
                            kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


# ============================================================
# Spectral Layer (Gabor)
# ============================================================

class SpectralLayer(nn.Module):
    """Learnable Gabor filter for frequency-selective temporal modeling.
    
    Acts as a parameterized band-pass filter that selectively amplifies
    narrow spectral bands relevant to abrupt storm intensification.
    """
    def __init__(self, T_in, T_out, weight_scale, alpha=1.0, beta=1.0, freq_multiplier=1.5):
        super().__init__()
        self.linear = nn.Linear(T_in, T_out)
        self.mu = nn.Parameter(2 * torch.rand(T_out, T_in) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((T_out,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        self.freq = nn.Parameter(torch.rand(T_out))
        self.freq_multiplier = freq_multiplier

    def forward(self, x):
        # x: (B, C, H, W, T_in) -> (B, C, H, W, T_out)
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.freq_multiplier * self.freq * self.linear(x)) * \
               torch.exp(-0.5 * D * self.gamma[None, :])


# ============================================================
# Temporal Layer (MLP)
# ============================================================

class TemporalLayer(nn.Module):
    """Nonlinear temporal projection for learning smooth trajectory dynamics."""
    def __init__(self, T_in, T_out, size_factor=1.0):
        super().__init__()
        hidden = int(T_out * size_factor)
        self.net = nn.Sequential(
            nn.Linear(T_in, hidden),
            nn.SELU(True),
            nn.Linear(hidden, T_out),
        )

    def forward(self, x):
        # x: (B, C, H, W, T_in) -> (B, C, H, W, T_out)
        return self.net(x)


# ============================================================
# Core Operator Block
# ============================================================

class LASTOCastBlock(nn.Module):
    """Single LASTOCast operator block.
    
    Comprises:
        1. Spectral Temporal Modeling  — dual-stream Gabor + MLP
        2. Spectro-Temporal Fusion     — learned per-pixel gating
        3. Spatio-Temporal Interaction  — joint spatial-temporal operator
        4. Residual Gabor connection    — high-frequency preservation
    """
    def __init__(self, T_in, T_out, dim, weight_scale, alpha, beta,
                 freq_multiplier, size_factor):
        super().__init__()
        self.T_out = T_out

        # --- Spectral Temporal Modeling ---
        self.spectral = SpectralLayer(T_in, T_out, weight_scale, alpha, beta, freq_multiplier)
        self.temporal = TemporalLayer(T_in, T_out, size_factor)

        # --- Spectro-Temporal Fusion ---
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

        # --- Spatio-Temporal Interaction ---
        self.spatial_temporal = nn.Sequential(
            ConvTransformBlock(dim * T_out, dim * T_out),
            ConvTransformBlock(dim * T_out, dim * T_out),
            nn.Conv2d(dim * T_out, dim * T_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # x: (B, T_in, C, H, W)

        # --- Spectral Temporal Modeling ---
        x_perm = x.permute(0, 2, 3, 4, 1)              # (B, C, H, W, T_in)
        alpha = self.spectral(x_perm).permute(0, 4, 1, 2, 3)  # (B, T_out, C, H, W)
        beta = self.temporal(x_perm).permute(0, 4, 1, 2, 3)   # (B, T_out, C, H, W)

        # --- Spectro-Temporal Fusion ---
        fused = torch.cat([alpha, beta], dim=2)          # (B, T_out, 2C, H, W)
        fused = fused.permute(0, 2, 1, 3, 4)            # (B, 2C, T_out, H, W)
        x = self.fusion(fused)                           # (B, C, T_out, H, W)
        x = x.permute(0, 2, 1, 3, 4)                    # (B, T_out, C, H, W)

        # --- Spatio-Temporal Interaction ---
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.spatial_temporal(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.T_out)

        # --- Residual Gabor Connection (high-frequency preservation) ---
        x = x + alpha

        return x


# ============================================================
# LASTOCast Neural Operator
# ============================================================

class LASTOCast(nn.Module):
    """Latent Spectro-Temporal Operator for Precipitation Nowcasting.
    
    Operates in the latent space of a pretrained autoencoder.
    Maps T_in input frames to T_out predicted frames.
    """
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale, alpha, beta, freq_multiplier, size_factor):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out

        # --- Lifting (Latent-to-Dynamics Extractor) ---
        self.lifting = nn.Sequential(
            ConvTransformBlock(in_dim, hidden_dim),
            ConvTransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # --- Core Operator Block ---
        self.operator = LASTOCastBlock(
            T_in, T_out, hidden_dim,
            weight_scale, alpha, beta, freq_multiplier, size_factor,
        )

        # --- Projection (Dynamics-to-Latent Projector) ---
        self.projection = nn.Sequential(
            ConvTransformBlock(hidden_dim, hidden_dim),
            ConvTransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, in_dim, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, T_in, C, H, W)

        # Lifting
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.lifting(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_in)

        # Core operator
        x = self.operator(x)

        # Projection
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.projection(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.T_out)

        return x


# ============================================================
# Training Wrapper
# ============================================================

class LASTOCastForecaster(nn.Module):
    """Training wrapper with loss computation."""
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 weight_scale, alpha, beta, freq_multiplier, size_factor,
                 total_steps, const_ratio):
        super().__init__()
        self.lastocast = LASTOCast(
            T_in, T_out, in_dim, hidden_dim,
            weight_scale, alpha, beta, freq_multiplier, size_factor,
        )
        self.T_in = T_in
        self.T_out = T_out
        self.mse = nn.MSELoss()
        self.itr = 0

    def forward(self, x, y=None, cmp_fft_loss=False):
        self.itr += 1
        return self.lastocast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            mse_loss = self.mse(xas, frames_gt)
            loss = {'total_loss': mse_loss}
            return xas, loss
        else:
            return xas, None


# ============================================================
# Model Factory
# ============================================================

def get_model(
    weight_scale,
    alpha,
    beta,
    freq_multiplier,
    size_factor,
    total_steps,
    const_ratio,
    img_channels=1,
    dim=64,
    T_in=5,
    T_out=20,
    **kwargs
):
    model = LASTOCastForecaster(
        T_in=T_in,
        T_out=T_out,
        in_dim=img_channels,
        hidden_dim=dim,
        weight_scale=weight_scale,
        alpha=alpha,
        beta=beta,
        freq_multiplier=freq_multiplier,
        size_factor=size_factor,
        total_steps=total_steps,
        const_ratio=const_ratio,
    )
    return model