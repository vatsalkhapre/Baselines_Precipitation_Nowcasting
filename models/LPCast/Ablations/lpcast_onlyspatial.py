"""
LPCast: Latent Precipitation Nowcasting

Architecture Overview:
    Input (B, T_in, C, H, W)
        → Latent Stem  : Latent channel expansion (C → hidden_dim)
        → PEM          : Precipitation Evolution Module
                             TemporalProjector  — T_in → T_out per spatial location
                             LatentEvolutionMixer (LEM) — The temporal and feature dimensions are flattened into the channel dimension before convolution, enabling cross-horizon and cross-feature interactions..
        → Projection   : Latent channel compression (hidden_dim → C), per frame
    Output (B, T_out, C, H, W)
"""

from torch import nn
from einops import rearrange

from utils.utilspp import RandomScheduling


# ============================================================
# Building Blocks
# ============================================================

class FeatureUnit(nn.Module):
    """Atomic feature extraction unit: Conv2d → GroupNorm → SiLU."""

    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super(FeatureUnit, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size,
                              padding=kernel_size // 2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class RFB(nn.Module):
    """Residual Feature Block (RFB).

    Two stacked FeatureUnits with a residual (skip) connection.
    Handles mismatched input/output dims via a 1×1 projection on the skip path.
    """

    def __init__(self, dim, dim_out, groups=8, kernel_size=3, padding_mode='zeros'):
        super().__init__()
        self.block1 = FeatureUnit(dim, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = FeatureUnit(dim_out, dim_out, groups=groups, kernel_size=kernel_size, padding_mode=padding_mode)
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
    """Build a lifting/projection path from a simple channel list."""
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
                RFB(
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
# Precipitation Evolution Module (PEM)
# ============================================================

class PEM(nn.Module):
    """Precipitation Evolution Module (PEM).

    Core forecasting block of LPCast. Evolves latent precipitation
    representations from T_in observed frames to T_out predicted frames via
    two sequential stages:

        1. TemporalProjector  — an MLP applied independently at each spatial
                                location to project T_in → T_out time steps.
        2. Latent Evolution Mixer (LEM) — jointly mixes future horizons,latent features, and 
                                          local spatial context over the projected latent representations.
    """

    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.t_out = t_out

        # Stage 1: temporal projection  (T_in → T_out) per spatial location
        self.temporal_projector = nn.Sequential(
            nn.Linear(t_in, int(t_out * size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out * size_factor), t_out),
        )

        # Stage 2: joint horizon-feature-spatial mixing over projected latent states
        self.lem = nn.Sequential(
            RFB(dim , dim ),
            RFB(dim , dim ),
            nn.Conv2d(dim , dim , kernel_size=3, padding=1),
        )

    def forward(self, x):
        # Temporal projection: (B, T_in, C, H, W) → (B, T_out, C, H, W)
        residual = self.temporal_projector(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        # Flatten time and feature dimensions into channels for joint horizon-feature-spatial mixing.: (B, T_out*C, H, W)
        x = rearrange(residual, 'b t c h w -> (b t) c h w')
        x = self.lem(x)

        # Restore temporal dimension and add residual from temporal projection
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.t_out)
        x = x + residual
        return x


# ============================================================
# LPCast Inner Network
# ============================================================

class LPCBackbone(nn.Module):
    """Inner network of LPCast.

    Orchestrates the full Latent Stem → PEM → Projection pipeline, mapping
    T_in latent input frames to T_out predicted latent frames.
    """

    def __init__(
        self,
        pre_seq_length,
        aft_seq_length,
        dim,
        hidden_dim,
        size_factor,
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

        # Latent Stem: expand latent channels independently per frame
        self.latent_stem, stem_out = build_channel_path(
            in_channels=dim,
            dims=lift_dims,
            groups=groups,
            padding_mode=padding_mode,
            last_kernel_size=1,
        )
        if stem_out != hidden_dim:
            raise ValueError(
                f"Latent stem must end at hidden_dim={hidden_dim}, but it ends at {stem_out}."
            )

        # PEM: Precipitation Evolution Module — temporal forecasting + spatial refinement
        self.pem = PEM(pre_seq_length, aft_seq_length, hidden_dim, size_factor)

        # Projection: compress back to latent channels per frame
        self.projection, proj_out = build_channel_path(
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
        # Latent Stem: process each frame independently  (B*T, C, H, W)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.latent_stem(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)

        # PEM: evolve T_in latent frames → T_out predicted latent frames
        x = self.pem(x)

        # Projection: compress features back to latent channels  (B*T_out, C, H, W)
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.projection(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return x


# ============================================================
# LPCast: Latent Precipitation Nowcasting
# ============================================================

class LPCast(nn.Module):
    """LPCast: Latent Precipitation Nowcasting.

    Training wrapper around the LPCBackbone inner network.
    Handles forward inference and loss computation via RandomScheduling.
    """

    def __init__(
        self,
        total_steps,
        const_ratio,
        pre_seq_length,
        aft_seq_length,
        input_dim,
        hidden_dim,
        size_factor=1,
        lift_dims=None,
        proj_dims=None,
        groups=8,
        padding_mode='zeros',
    ):
        super(LPCast, self).__init__()
        self.backbone = LPCBackbone(
            pre_seq_length,
            aft_seq_length,
            input_dim,
            hidden_dim,
            size_factor,
            lift_dims=lift_dims,
            proj_dims=proj_dims,
            groups=groups,
            padding_mode=padding_mode,
        )
        self.criterion = RandomScheduling(total_steps, 1, const_ratio)

    def forward(self, x):  # x: (B, T_in, C, H, W)
        return self.backbone(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        xas = self(frames_in)
        if compute_loss:
            facl_loss = self.criterion(xas, frames_gt)
            loss = {'total_loss': facl_loss}
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
    lift_dims=[64,64,64], 
    proj_dims=[64,64,64,4],
    **kwargs
):
    model = LPCast(
        total_steps, const_ratio,
        pre_seq_length=T_in,
        aft_seq_length=T_out,
        input_dim=img_channels,
        hidden_dim=dim,
        size_factor=mlp_size_factor,
        lift_dims=lift_dims, 
        proj_dims=proj_dims
    )
    return model