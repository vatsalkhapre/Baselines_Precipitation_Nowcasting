"""
Incremental Ablation — Step 3.5: MLP + Gabor + Wavelet + AFNO only
Adds AFNO-only spatio-temporal interaction after temporal modeling.
No depthwise conv, no pointwise/channel mixing.
conv_spectral = ResneSpectralBlock x2 (AFNO only) + standalone AFNO.
Lifting and Projection present.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from pytorch_wavelets import DWTForward, DWTInverse
from utils.utilspp import RandomScheduling


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size,
                              padding=kernel_size // 2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class TransformBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.skip   = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.skip(x)


class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale,
                 alpha=1.0, beta=1.0, freq_multiplier=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu     = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma  = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        self.freq            = nn.Parameter(torch.rand(out_features))
        self.freq_multiplier = freq_multiplier

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.freq_multiplier * self.freq * self.linear(x)) * \
               torch.exp(-0.5 * D * self.gamma[None, :])


class BandTemporalStream(nn.Module):
    def __init__(self, t_in, t_out, dim, weight_scale, alpha, beta,
                 freq_multiplier, size_factor=1.0):
        super().__init__()
        self.gabor  = GaborLayer(t_in, t_out, weight_scale, alpha, beta, freq_multiplier)
        self.mlp    = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=1)

    def forward(self, x):
        gabor_out = self.gabor(x)
        mlp_out   = self.mlp(x)
        fused = torch.cat([gabor_out, mlp_out], dim=1)
        fused = fused.permute(0, 1, 4, 2, 3)
        fused = self.fusion(fused)
        return gabor_out, mlp_out, fused


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=1, sparsity_threshold=0.01,
                 hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0
        self.hidden_size        = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks         = num_blocks
        self.block_size         = hidden_size // num_blocks
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
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros_like(o1_real)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros_like(o2_real)

        total_modes = N // 2 + 1
        kept_modes  = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) -
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + self.b1[0]
        )
        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) +
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + self.b1[1]
        )
        o2_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) -
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + self.b2[0]
        )
        o2_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) +
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.type(dtype)
        return x + bias


class SpectralBlock_AFNOOnly(nn.Module):
    """
    AFNO only — no dw_spatial, no pw channel mixing.
    Input/output: (B, H, W, C)
    """
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor, groups=8):
        super().__init__()
        self.proj = AFNO2D(dim, num_blocks, sparsity_threshold,
                           hidden_size_factor=hidden_size_factor)
        self.norm = nn.GroupNorm(groups, dim)
        self.act  = nn.SiLU()

    def forward(self, x):
        # x: (B, H, W, C)
        x_spec = self.proj(x)                        # AFNO, residual inside
        x_spec = x_spec.permute(0, 3, 1, 2)         # (B, C, H, W)
        x_spec = self.norm(x_spec)
        x_spec = self.act(x_spec)
        return x_spec.permute(0, 2, 3, 1)           # (B, H, W, C)


class ResSpectralBlock_AFNOOnly(nn.Module):
    def __init__(self, dim, num_blocks, sparsity_threshold, hidden_size_factor):
        super().__init__()
        self.block1   = SpectralBlock_AFNOOnly(dim, num_blocks, sparsity_threshold, hidden_size_factor)
        self.block2   = SpectralBlock_AFNOOnly(dim, num_blocks, sparsity_threshold, hidden_size_factor)
        self.res_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        # x: (B, H, W, C)
        h   = self.block1(x)
        h   = self.block2(h)
        res = self.res_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return h + res


class WaveletGaborBlock(nn.Module):
    """
    Step 3.5: Wavelet + Gabor+MLP + AFNO-only spatio-temporal interaction.
    No dw_spatial conv, no pw channel mixing.
    """
    def __init__(self, t_in, t_out, dim,
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor=1.0,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.dim     = dim
        self.level   = level
        self.hf_mode = hf_mode

        self.dwt  = DWTForward(J=level, wave=wave, mode='zero')
        self.idwt = DWTInverse(wave=wave, mode='zero')

        self.stream_ll = BandTemporalStream(
            t_in, t_out, dim,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low, size_factor,
        )
        if hf_mode == 'shared':
            self.stream_hf = BandTemporalStream(
                t_in, t_out, 3 * dim,
                weight_scale_high, alpha_high, beta_high, freq_multiplier_high, size_factor,
            )
        else:
            self.hf_streams = nn.ModuleList()
            for i in range(level):
                freq_mid     = (freq_multiplier_low + freq_multiplier_high) / 2
                alpha_interp = i / (level - 1) if level > 1 else 0
                freq_i       = freq_multiplier_high * (1 - alpha_interp) + freq_mid * alpha_interp
                self.hf_streams.append(BandTemporalStream(
                    t_in, t_out, 3 * dim,
                    weight_scale_high, alpha_high, beta_high, freq_i, size_factor,
                ))

        # AFNO only — no dw conv, no pw
        self.conv_spectral = nn.Sequential(
            ResSpectralBlock_AFNOOnly(dim * t_out, num_blocks, sparsity_threshold, hidden_size_factor),
            ResSpectralBlock_AFNOOnly(dim * t_out, num_blocks, sparsity_threshold, hidden_size_factor),
            AFNO2D(dim * t_out, num_blocks, sparsity_threshold, hidden_size_factor=hidden_size_factor),
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        x_flat = rearrange(x, 'b t c h w -> (b t) c h w')
        ll, hf_list = self.dwt(x_flat)

        ll_t = rearrange(ll, '(b t) c h w -> b c h w t', t=T)
        ll_gabor, ll_mlp, ll_fused = self.stream_ll(ll_t)

        hf_gabor_list = []
        hf_fused_list = []
        for i, hf in enumerate(hf_list):
            hf_t = rearrange(hf, '(b t) c n h w -> b (c n) h w t', t=T)
            if self.hf_mode == 'shared':
                hf_gabor, hf_mlp, hf_fused = self.stream_hf(hf_t)
            else:
                hf_gabor, hf_mlp, hf_fused = self.hf_streams[i](hf_t)
            hf_gabor_list.append(hf_gabor)
            hf_fused_list.append(hf_fused)

        # IDWT fused
        ll_recon = rearrange(ll_fused, 'b c t h w -> (b t) c h w')
        hf_recon_list = [
            rearrange(hf, 'b (c n) t h w -> (b t) c n h w', n=3)
            for hf in hf_fused_list
        ]
        reconstructed = self.idwt((ll_recon, hf_recon_list))[..., :H, :W]
        reconstructed = rearrange(reconstructed, '(b t) c h w -> b t c h w', t=self.t_out)

        # IDWT gabor residual
        ll_gabor_flat = rearrange(ll_gabor, 'b c h w t -> (b t) c h w')
        hf_gabor_flat_list = [
            rearrange(hf, 'b (c n) h w t -> (b t) c n h w', n=3)
            for hf in hf_gabor_list
        ]
        gabor_residual = self.idwt((ll_gabor_flat, hf_gabor_flat_list))[..., :H, :W]
        gabor_residual = rearrange(gabor_residual, '(b t) c h w -> b t c h w', t=self.t_out)

        # AFNO-only spatio-temporal interaction
        x_st = rearrange(reconstructed, 'b t c h w -> b h w (t c)')
        x_st = self.conv_spectral(x_st)
        x_st = rearrange(x_st, 'b h w (t c) -> b t c h w', t=self.t_out)

        return x_st + gabor_residual


class WaveletLASTOCast(nn.Module):
    def __init__(self, T_in, T_out, in_dim, hidden_dim,
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor=1.0,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.T_in  = T_in
        self.T_out = T_out
        self.lifting = nn.Sequential(
            TransformBlock(in_dim, hidden_dim),
            TransformBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.operator = WaveletGaborBlock(
            T_in, T_out, hidden_dim,
            num_blocks, sparsity_threshold, hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,
        )
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
                 num_blocks, sparsity_threshold, hidden_size_factor,
                 weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
                 weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
                 k_spatial, size_factor, total_steps, const_ratio,
                 wave='haar', level=1, hf_mode='shared'):
        super().__init__()
        self.lastocast = WaveletLASTOCast(
            T_in, T_out, in_dim, hidden_dim,
            num_blocks, sparsity_threshold, hidden_size_factor,
            weight_scale_low, alpha_low, beta_low, freq_multiplier_low,
            weight_scale_high, alpha_high, beta_high, freq_multiplier_high,
            k_spatial, size_factor, wave, level, hf_mode,
        )
        self.T_in   = T_in
        self.T_out  = T_out
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        self.itr    = 0

    def forward(self, x, y=None, cmp_fft_loss=False):
        self.itr += 1
        return self.lastocast(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            falfcl_loss = self.falfcl(xas, frames_gt)
            return xas, {'total_loss': falfcl_loss}
        return xas, None


def get_model(
    num_blocks=1, sparsity_threshold=0.01, hidden_size_factor=1,
    weight_scale_low=1.0, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=1.0, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
    k_spatial=3, size_factor=1.0,
    total_steps=50000, const_ratio=0.1,
    img_channels=1, dim=64,
    T_in=5, T_out=20,
    wave='haar', wavelet_level=1, hf_mode='shared',
    input_shape=(128, 128), **kwargs
):
    return WaveletLASTOCastForecaster(
        T_in=T_in, T_out=T_out,
        in_dim=img_channels, hidden_dim=dim,
        num_blocks=num_blocks, sparsity_threshold=sparsity_threshold,
        hidden_size_factor=hidden_size_factor,
        weight_scale_low=weight_scale_low, alpha_low=alpha_low,
        beta_low=beta_low, freq_multiplier_low=freq_multiplier_low,
        weight_scale_high=weight_scale_high, alpha_high=alpha_high,
        beta_high=beta_high, freq_multiplier_high=freq_multiplier_high,
        k_spatial=k_spatial, size_factor=size_factor,
        total_steps=total_steps, const_ratio=const_ratio,
        wave=wave, level=wavelet_level, hf_mode=hf_mode,
    )
