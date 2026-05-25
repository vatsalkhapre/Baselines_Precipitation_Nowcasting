
"""
AmpliNet: Amplitude-Based Spatio-Temporal Operator for Precipitation Nowcasting

Architecture Overview:
    Input (B, T_in, C, H, W)
        → Lifting (input-to-hidden feature extractor)
        → AmpCell Layers (temporal amplitude projection + spatial conv)
        → Projection (hidden-to-output feature projector)
    Output (B, T_out, C, H, W)

This version keeps lifting/projection configurable through simple
channel schedules:

    lift_dims = [16, 32, 64]
    proj_dims = [32, 16, 4]

and keeps AmpCell kernel sizes configurable as a list:

    conv_kernel_sizes = [7, 5, 3]
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
        self.proj = nn.Conv2d(
            dim,
            dim_out,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode=padding_mode,
        )
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


def build_channel_path(
    in_channels,
    dims,
    groups=8,
    padding_mode='zeros',
    last_kernel_size=1,
):
    """
    Build a lifting/projection path from a simple channel list.

    The convention is:
      - all intermediate stages use ResnetBlock
      - the last stage uses a plain Conv2d
    """
    if dims is None or len(dims) == 0:
        return nn.Identity(), in_channels

    layers = []
    prev = in_channels

    for idx, out_ch in enumerate(dims):
        is_last = idx == len(dims) - 1
        if is_last:
            layers.append(
                nn.Conv2d(
                    prev,
                    out_ch,
                    kernel_size=last_kernel_size,
                    padding=last_kernel_size // 2,
                    padding_mode=padding_mode,
                )
            )
        else:
            layers.append(
                ResnetBlock(
                    prev,
                    out_ch,
                    groups=groups,
                    kernel_size=3,
                    padding_mode=padding_mode,
                )
            )
        prev = out_ch

    return nn.Sequential(*layers), prev


# ============================================================
# Core Operator Block
# ============================================================

class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, size_factor=1.0, conv_kernel_size_1=3, conv_kernel_size_2=3, conv_kernel_size_3=3):
        super().__init__()
        self.t_out = t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )
        self.conv = nn.Sequential(
            ResnetBlock(dim * t_out, dim * t_out, kernel_size=conv_kernel_size_1),
            ResnetBlock(dim * t_out, dim * t_out, kernel_size=conv_kernel_size_2),
            nn.Conv2d(dim * t_out, dim * t_out, kernel_size=conv_kernel_size_3, padding=conv_kernel_size_3 // 2),
        )

    def forward(self, x):
        residual = self.tmlp(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        x = rearrange(residual, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        x = x + residual
        return x


# ============================================================
# AmpliNet Neural Operator
# ============================================================

class AmpliNet(nn.Module):
    """
    Amplitude projection network.

    Channel schedules are given by simple lists:
      lift_dims = [16, 32, 64]
      proj_dims = [32, 16, 4]
    """

    def __init__(
        self,
        pre_seq_length,
        aft_seq_length,
        dim,
        hidden_dim,
        size_factor,
        conv_kernel_sizes,
        lift_dims=None,
        proj_dims=None,
        groups=8,
        padding_mode='zeros',
    ):
        super().__init__()
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length

        # Default to the original 3-stage arrangement if not provided.
        if lift_dims is None:
            lift_dims = [hidden_dim, hidden_dim, hidden_dim]
        if proj_dims is None:
            proj_dims = [hidden_dim, hidden_dim, dim]

        if len(lift_dims) == 0:
            raise ValueError("lift_dims must contain at least one channel size.")
        if len(proj_dims) == 0:
            raise ValueError("proj_dims must contain at least one channel size.")

        # Enforce the intended endpoint consistency.
        if lift_dims[-1] != hidden_dim:
            raise ValueError(
                f"lift_dims must end at hidden_dim={hidden_dim}, but got {lift_dims[-1]}."
            )
        if proj_dims[0] != hidden_dim:
            raise ValueError(
                f"proj_dims must start from hidden_dim={hidden_dim}, but got {proj_dims[0]}."
            )
        if proj_dims[-1] != dim:
            raise ValueError(
                f"proj_dims must end at img_channels={dim}, but got {proj_dims[-1]}."
            )

        # --- Lifting (Input-to-Hidden Feature Extractor) ---
        self.convin, lift_out = build_channel_path(
            in_channels=dim,
            dims=lift_dims,
            groups=groups,
            padding_mode=padding_mode,
            last_kernel_size=1,
        )
        if lift_out != hidden_dim:
            raise ValueError(
                f"Lifting path must end at hidden_dim={hidden_dim}, but it ends at {lift_out}."
            )

        conv_kernel_size_1, conv_kernel_size_2, conv_kernel_size_3 = conv_kernel_sizes

        # --- Core Operator Block ---
        self.ampcell = AmpCell(
            pre_seq_length,
            aft_seq_length,
            hidden_dim,
            size_factor,
            conv_kernel_size_1,
            conv_kernel_size_2,
            conv_kernel_size_3,
        )

        # --- Projection (Hidden-to-Output Feature Projector) ---
        self.convout, proj_out = build_channel_path(
            in_channels=hidden_dim,
            dims=proj_dims,
            groups=groups,
            padding_mode=padding_mode,
            last_kernel_size=1,
        )
        if proj_out != dim:
            raise ValueError(
                f"Projection path must end at img_channels={dim}, but it ends at {proj_out}."
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
    def __init__(
        self,
        total_steps,
        const_ratio,
        pre_seq_length,
        aft_seq_length,
        input_dim,
        hidden_dim,
        conv_kernel_sizes,
        size_factor=1,
        lift_dims=None,
        proj_dims=None,
        groups=8,
        padding_mode='zeros',
    ):
        super(AlphaPre_Amplinet, self).__init__()
        self.amplinet = AmpliNet(
            pre_seq_length,
            aft_seq_length,
            input_dim,
            hidden_dim,
            size_factor,
            conv_kernel_sizes,
            lift_dims=lift_dims,
            proj_dims=proj_dims,
            groups=groups,
            padding_mode=padding_mode,
        )
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
    conv_kernel_sizes=(3, 3, 3),
    lift_dims=None,
    proj_dims=None,
    **kwargs
):
    model = AlphaPre_Amplinet(
        total_steps,
        const_ratio,
        pre_seq_length=T_in,
        aft_seq_length=T_out,
        input_dim=img_channels,
        hidden_dim=dim,
        conv_kernel_sizes=conv_kernel_sizes,
        size_factor=mlp_size_factor,
        lift_dims=lift_dims,
        proj_dims=proj_dims,
    )
    return model
