"""
AmpliNet: Amplitude-Based Spatio-Temporal Operator for Precipitation Nowcasting

Architecture Overview:
    Input (B, T_in, C, H, W)
        → Lifting (input-to-hidden feature extractor)
        → AmpCell Layers (temporal amplitude projection + spatial conv)
        → Projection (hidden-to-output feature projector)
    Output (B, T_out, C, H, W)
"""

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from utils.utilspp import RandomScheduling


# ============================================================
# Building Blocks
# ============================================================

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros', groupnorm=True):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding=kernel_size // 2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):  # 'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, dim_out, 3, padding=1)
    )


def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )


# ============================================================
# Core Operator Block
# ============================================================

class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.conv = nn.Sequential(
            ResnetBlock(dim * t_out, dim * t_out),
            ResnetBlock(dim * t_out, dim * t_out),
            nn.Conv2d(dim * t_out, dim * t_out, kernel_size=3, padding=1),
        )

    def forward(self, x):
        residual = self.tmlp(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = residual
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        x = x + residual
        return x


# ============================================================
# AmpliNet Neural Operator
# ============================================================

class AmpliNet(nn.Module):
    """Amplitude projection network.

    Maps T_in input frames to T_out predicted frames via stacked AmpCell layers.
    """
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=1, mlp_ratio=2):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim

        # --- Lifting (Input-to-Hidden Feature Extractor) ---
        self.convin = nn.Sequential(
            ResnetBlock(dim, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # --- Core Operator Blocks ---
        self.ampcell = AmpCell(pre_seq_length , aft_seq_length, hidden_dim)


        # --- Projection (Hidden-to-Output Feature Projector) ---
        self.convout = nn.Sequential(
            ResnetBlock(hidden_dim, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, dim, kernel_size=1),
        )

    def forward(self, x):
        # Lifting
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)

        # Core operator
        x = self.ampcell(x)

        # Projection
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convout(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return x


# ============================================================
# Training Wrapper
# ============================================================

class AlphaPre_Amplinet(nn.Module):
    """Training wrapper with loss computation."""
    def __init__(self, total_steps, const_ratio, pre_seq_length, aft_seq_length, input_shape, input_dim,
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
        self.criterion = nn.MSELoss()
        self.itr = 0
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate = self.amp_weight / self.aweight_stop_steps

        h, w = input_shape
        spec_mask = torch.zeros(h, w // 2 + 1)
        spec_mask[..., :spec_num, :spec_num] = 1.
        spec_mask[..., -spec_num:, :spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)

    def forward(self, x, y, cmp_fft_loss=False):  # x: [b,t,c,h,w]
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
                self.amp_weight = 0.

            loss = 0.

            # frames_fft = torch.fft.rfft2(frames_gt)
            # frames_abs = torch.abs(frames_fft)
            # xas_fft = torch.fft.rfft2(xas)
            # xas_abs = torch.abs(xas_fft)
            # amp_loss = self.criterion(xas_abs, frames_abs)
            # loss += self.amp_weight*amp_loss
            mse_loss = self.criterion(xas, frames_gt)
            loss = {'total_loss': mse_loss}
            return xas, loss
        else:
            return xas, None


# ============================================================
# Model Factory
# ============================================================

def get_model(
    total_steps,
    const_ratio,
    img_channels=1,
    dim=64,
    T_in=5,
    T_out=20,
    input_shape=(128, 128),
    n_layers=3,
    spec_num=20,
    pha_weight=0.01,
    anet_weight=0.1,
    amp_weight=0.01,
    aweight_stop_steps=10000,
    **kwargs
):
    model = AlphaPre_Amplinet(
        total_steps, const_ratio,
        pre_seq_length=T_in,
        aft_seq_length=T_out,
        input_shape=input_shape,
        input_dim=img_channels,
        hidden_dim=dim,
        n_layers=n_layers,
        spec_num=spec_num,
        pha_weight=pha_weight,
        anet_weight=anet_weight,
        amp_weight=amp_weight,
        aweight_stop_steps=aweight_stop_steps,
    )
    return model