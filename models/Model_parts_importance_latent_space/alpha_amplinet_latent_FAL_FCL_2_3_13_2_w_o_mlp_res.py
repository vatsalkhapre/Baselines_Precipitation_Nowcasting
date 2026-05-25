"""
AmpliNet: Amplitude-Based Spatio-Temporal Operator for Precipitation Nowcasting

Architecture Overview:
    Input (B, T_in, C, H, W)
        → Lifting (input-to-hidden feature extractor)
        → AmpCell Layers (temporal amplitude projection + spatial conv)
        → Projection (hidden-to-output feature projector)
    Output (B, T_out, C, H, W)
"""

from torch import nn
from einops import rearrange

from utils.utilspp import RandomScheduling


# ============================================================
# Building Blocks
# ============================================================

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size,
                              padding=kernel_size // 2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


# ============================================================
# Core Operator Block
# ============================================================

class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.t_out = t_out
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
        x = rearrange(residual, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        # x = x + residual
        return x


# ============================================================
# AmpliNet Neural Operator
# ============================================================

class AmpliNet(nn.Module):
    """Amplitude projection network.

    Maps T_in input frames to T_out predicted frames via an AmpCell layer.
    """
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, size_factor):
        super().__init__()
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length

        # --- Lifting (Input-to-Hidden Feature Extractor) ---
        self.convin = nn.Sequential(
            ResnetBlock(dim, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        # --- Core Operator Block ---
        self.ampcell = AmpCell(pre_seq_length, aft_seq_length, hidden_dim, size_factor)

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
    def __init__(self, total_steps, const_ratio, pre_seq_length, aft_seq_length,
                 input_dim, hidden_dim, size_factor=1):
        super(AlphaPre_Amplinet, self).__init__()
        self.amplinet = AmpliNet(pre_seq_length, aft_seq_length, input_dim, hidden_dim, size_factor)
        self.criterion = RandomScheduling(total_steps, 1, const_ratio)

    def forward(self, x):  # x: [b, t, c, h, w]
        return self.amplinet(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in)
        if compute_loss:
            falfcl_loss = self.criterion(xas, frames_gt)
            loss = {'total_loss': falfcl_loss}
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
    mlp_size_factor=1.0,
    **kwargs
):
    model = AlphaPre_Amplinet(
        total_steps, const_ratio,
        pre_seq_length=T_in,
        aft_seq_length=T_out,
        input_dim=img_channels,
        hidden_dim=dim,
        size_factor=mlp_size_factor,
    )
    return model