"""WADEPre baseline (Wavelet Approximation/Detail decomposition + Refiner).

Ported from the original PyTorch-Lightning implementation to a plain
``torch.nn.Module`` so it can be driven by ``run_alphapre_convlstm.py``.

The Lightning-specific plumbing (``save_hyperparameters``, ``self.log`` logging,
``training_step``/``validation_step``, ``configure_optimizers`` and the
``on_load_checkpoint`` hook) has been removed -- the accelerate-based runner in
``run_alphapre_convlstm.py`` owns the optimizer, scheduler, logging and
checkpointing. The model logic (forward pass + loss terms) is unchanged.

Architectural note: WADEPre is a *same-length* spatio-temporal model -- it maps
``timesteps`` input frames to ``timesteps`` output frames (the FPN input
channels, temporal MLPs and Refiner group-norms are all tied to a single
``timesteps`` value). We therefore run it in an autoregressive
``T_in -> T_in`` setup:

* build the model with ``timesteps = T_in`` (=5);
* **train** on a single step -- predict the next ``T_in`` frames from the
  ``T_in`` observed frames, supervised against the first ``T_in`` ground-truth
  future frames (``frames_gt[:, :T_in]``);
* **infer** autoregressively -- feed the predicted block back in and roll out
  ``ceil(T_out / T_in)`` times (4x for 5 -> 20), then trim to ``T_out``.

All of this lives inside :meth:`predict`, so ``run_alphapre_convlstm.py`` needs
no knowledge of the rollout.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .Approximation import ApproximationNetwork
from .Detail import DetailNetwork
from .Refiner import Refiner
from .utils.wavelet_transform import WaveletTransform, WaveletCoeffDict
from .utils.zncc import zncc


class WADEPre(nn.Module):
    def __init__(
        self,
        # general params
        timesteps: int,
        spatial_size: int,
        t_out: int = None,
        dropout_rate: float = 0.1,
        # detail network params
        detail_idr_dim: int = 32,
        detail_feature_channel: int = 64,
        detail_layer_channels: list = [64, 128, 256],
        detail_num_blocks: int = 4,
        # approximation network params
        approx_hidden_size: int = 128,
        approx_cells: int = 3,
        # refine mixer params
        refine_hidden_dim: int = 120,
        # wavelet params
        wavelet_name: str = "bior2.4",
        wavelet_level: int = 3,
        # loss weights
        loss_a_weight: float = 1.0,
        loss_a_constant_weight: float = 0.15,
        loss_a_stop_step: int = 5000,
        loss_d_weight: float = 1.0,
        loss_recon_mean_weight: float = 0.1,
    ):
        super().__init__()

        self.timesteps = timesteps
        self.spatial_size = spatial_size
        # forecast horizon and number of autoregressive rollout steps at
        # inference time (train stays single-step: T_in -> T_in).
        self.t_out = t_out if t_out is not None else timesteps
        self.n_rollout = math.ceil(self.t_out / self.timesteps)

        # ---- loss schedule params ----
        self.a_weight = loss_a_weight
        self.a_weight_decay = (loss_a_weight - loss_a_constant_weight) / loss_a_stop_step
        self.loss_a_constant_weight = loss_a_constant_weight
        self.loss_a_stop_step = loss_a_stop_step
        self.d_weight = loss_d_weight
        self.loss_recon_mean_weight = loss_recon_mean_weight

        # training-step counter for the amplitude-weight decay schedule
        # (replaces Lightning's self.global_step). Kept as a plain python int
        # so it does not enter the state_dict, matching the alphapre baseline.
        self.itr = 0

        self.detail_network = DetailNetwork(
            fpn_time=timesteps,
            idr_dim=detail_idr_dim,
            feature_channel=detail_feature_channel,
            layer_channels=detail_layer_channels,
            num_blocks=(
                detail_num_blocks
                if isinstance(detail_num_blocks, (list, tuple))
                else [detail_num_blocks] * len(detail_layer_channels)
            ),
            dropout_rate=dropout_rate,
        )

        self.approx_network = ApproximationNetwork(
            hidden_size=approx_hidden_size,
            timesteps=timesteps,
            cell_numbers=approx_cells,
            dropout_rate=dropout_rate,
        )

        self.refine_mixer = Refiner(
            time_steps=timesteps,
            hidden_dim=refine_hidden_dim,
            dropout_rate=dropout_rate,
        )

        self.wavelet_transform = WaveletTransform(
            wavelet=wavelet_name, level=wavelet_level, mode="reflect"
        )

        # lazily-built sub-layers (encoder/decoder/fpn) are created here
        self.init_model()

    def init_model(self) -> None:
        # dummy run to instantiate the lazily-built layers inside the
        # detail/approximation networks (they infer coefficient sizes here).
        x = torch.randn(
            2,  # batch size, 2 is enough for a dummy run
            self.timesteps,
            self.spatial_size,
            self.spatial_size,
        )
        self.detail_network.dummy_run(data=x, wavelet=self.wavelet_transform)
        self.approx_network.dummy_run(data=x, wavelet=self.wavelet_transform)

    def forward(self, x: torch.Tensor):
        """x: (B, T, H, W) with T == self.timesteps."""

        # Y_D, D
        d_reconstruction, d_coeff = self.detail_network.forward(
            x, wavelet=self.wavelet_transform
        )

        # Y_A, A
        a_reconstruction, a_coeff = self.approx_network.forward(
            x, wavelet=self.wavelet_transform
        )

        # Y_AD, set A first
        ad_coeff: WaveletCoeffDict = {"A": a_coeff["A"]}

        # Y_AD, set D coefficients
        for l in range(1, self.wavelet_transform.level + 1):
            level_key = f"D{l}"
            level_details = d_coeff[level_key]  # Tensor shaped (B, T, 3, H, W)

            if level_details.shape[2] != 3:
                raise RuntimeError(
                    f"Expected 3 detail channels at level {l}, got shape {level_details.shape}"
                )

            ad_coeff[level_key] = level_details

        # Y_AD
        ad_reconstruction = self.wavelet_transform.reverse(ad_coeff)

        # Refiner
        refined_out = self.refine_mixer.forward(
            AD_guide=ad_reconstruction,
            A_guide=a_reconstruction,
            D_guide=d_reconstruction,
            last_frame=x[:, -1:, :, :],
        )

        return refined_out, {
            "d_rec": d_reconstruction,
            "a_rec": a_reconstruction,
            "ad_rec": ad_reconstruction,
            "d_coeff": d_coeff,
            "a_coeff": a_coeff,
            "ad_coeff": ad_coeff,
            # Only refined_out is the final forecast; the rest feed the loss.
            "refined_out": refined_out,
        }

    def _compute_loss(self, x: dict, truth: torch.Tensor) -> dict:
        """Reproduces the original WADEPre loss.

        x     : the details dict returned by forward()
        truth : (B, T, H, W) ground-truth frames
        """
        truth_wave: WaveletCoeffDict = self.wavelet_transform.transform(truth)

        # amplitude-coefficient weight decays linearly then holds constant
        if self.itr < self.loss_a_stop_step:
            a_weight = self.a_weight - self.itr * self.a_weight_decay
        else:
            a_weight = self.loss_a_constant_weight
        self.itr += 1

        # L_pred
        main_recon = F.mse_loss(x["refined_out"], truth)

        # L_A (zero-normalised cross-correlation on approximation coeffs)
        a_coeff_loss = zncc(x["a_coeff"]["A"], truth_wave["A"])

        # L_D (per-level detail-coefficient MSE, weighted by 1/2^l)
        d_coeff_loss = 0.0
        for l in range(1, self.wavelet_transform.level + 1):
            d_coeff_loss += F.mse_loss(
                x["d_coeff"][f"D{l}"], truth_wave[f"D{l}"]
            ) * (1.0 / (2 ** l))

        # L_Mixed (mean of the three reconstructions vs. truth)
        reconstruction_mean = (x["ad_rec"] + x["a_rec"] + x["d_rec"]) / 3
        reconstruction_mean_loss = F.mse_loss(reconstruction_mean, truth)

        total_loss = (
            main_recon
            + a_weight * a_coeff_loss
            + self.d_weight * d_coeff_loss
            + self.loss_recon_mean_weight * reconstruction_mean_loss
        )

        return {
            "total_loss": total_loss,
            "recon": main_recon,
            "a_coeff": a_coeff_loss,
            "d_coeff": d_coeff_loss,
            "recon_mean": reconstruction_mean_loss,
        }

    def _to_input_block(self, frames_in: torch.Tensor) -> torch.Tensor:
        """(B, T_in, C=1, H, W) -> (B, timesteps, H, W).

        Squeezes the channel dim and matches the temporal length to
        ``self.timesteps`` (take the most recent frames if longer, repeat the
        last frame if shorter). In the standard 5 -> 5 setup this is a no-op
        beyond the channel squeeze.
        """
        C = frames_in.shape[2]
        if C != 1:
            raise ValueError(
                f"WADEPre operates on single-channel radar frames, got C={C}"
            )
        x = frames_in[:, :, 0, :, :]                    # (B, T_in, H, W)
        T_in = x.shape[1]
        if T_in > self.timesteps:
            x = x[:, -self.timesteps:, :, :]            # keep most recent frames
        elif T_in < self.timesteps:
            pad = x[:, -1:, :, :].repeat(1, self.timesteps - T_in, 1, 1)
            x = torch.cat([x, pad], dim=1)
        return x

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        """Runner-facing interface.

        frames_in : (B, T_in, C, H, W)
        frames_gt : (B, T_out, C, H, W)   (only needed when compute_loss=True)

        Training  (compute_loss=True): single-step T_in -> T_in prediction,
            supervised against the first ``timesteps`` future frames. Returns
            pred shaped (B, timesteps, C, H, W).
        Inference (compute_loss=False): autoregressive rollout to build the full
            forecast. Returns pred shaped (B, T_out, C, H, W).
        """
        x = self._to_input_block(frames_in)             # (B, timesteps, H, W)

        if compute_loss:
            if frames_gt is None:
                raise ValueError("frames_gt is required when compute_loss=True")
            refined_out, details = self.forward(x)      # (B, timesteps, H, W)
            # supervise against the next `timesteps` ground-truth frames
            truth = frames_gt[:, : self.timesteps, 0, :, :]   # (B, timesteps, H, W)
            loss = self._compute_loss(details, truth)
            return refined_out.unsqueeze(2), loss       # (B, timesteps, 1, H, W)

        # ---- autoregressive rollout: feed each predicted block back in ----
        preds = []
        cur = x
        for _ in range(self.n_rollout):
            out, _ = self.forward(cur)                  # (B, timesteps, H, W)
            preds.append(out)
            cur = out
        full = torch.cat(preds, dim=1)[:, : self.t_out, :, :]   # (B, T_out, H, W)
        return full.unsqueeze(2), None                  # (B, T_out, 1, H, W)


def _safe_refine_hidden_dim(time_steps: int, target: int = 576) -> int:
    """Smallest multiple of lcm(time_steps, 8) closest to `target`.

    The Refiner ties several GroupNorms to `time_steps` (min(32, T) groups and
    `groups = time_steps`) and its ResnetBlocks default to 8 groups, so the
    hidden dim must be divisible by lcm(time_steps, 8) and by time_steps.
    """
    m = math.lcm(time_steps, 8)
    k = max(1, round(target / m))
    return k * m


def get_model(
    input_shape=(128, 128),
    T_in: int = 5,
    T_out: int = 20,
    img_channels: int = 1,
    dropout_rate: float = 0.1,
    # detail network            (defaults from models/WADEPre/train.py)
    detail_idr_dim: int = 64,
    detail_feature_channel: int = 128,
    detail_layer_channels=(64, 128, 256),
    detail_num_blocks: int = 4,
    # approximation network         (paper: "channel dimension is set to 256")
    approx_hidden_size: int = 256,
    approx_cells: int = 3,
    # refiner: paper reports hidden_dim=576, valid as-is only when
    # timesteps == 6 (WADEPre's native horizon). Refiner.py asserts
    # hidden_dim % time_steps == 0 and ties several GroupNorms to
    # lcm(time_steps, 8), so for our timesteps=T_in=5 autoregressive setup we
    # snap to the nearest valid multiple of lcm(time_steps, 8) instead
    # (560 for T_in=5; recovers the paper's exact 576 for T_in=6).
    refine_hidden_dim=None,
    # wavelet (WADEPre native defaults; not the DAWNCast --wave args)
    wavelet_name: str = "bior2.4",
    wavelet_level: int = 3,
    # loss weights              (defaults from models/WADEPre/train.py)
    loss_a_weight: float = 0.1,
    loss_a_constant_weight: float = 0.01,
    loss_a_stop_step: int = 3000,
    loss_d_weight: float = 0.05,
    loss_recon_mean_weight: float = 0.005,
    **kwargs,
):
    if img_channels != 1:
        raise ValueError("WADEPre only supports single-channel radar frames")
    if input_shape[0] != input_shape[1]:
        raise ValueError(f"WADEPre expects a square input, got {input_shape}")

    # same-length model runs T_in -> T_in and rolls out to T_out at inference
    timesteps = T_in
    if refine_hidden_dim is None:
        refine_hidden_dim = timesteps * 96
    else:
        # keep whatever was passed but ensure it satisfies the group-norm rules
        refine_hidden_dim = _safe_refine_hidden_dim(timesteps, target=refine_hidden_dim)

    model = WADEPre(
        timesteps=timesteps,
        spatial_size=input_shape[0],
        t_out=T_out,
        dropout_rate=dropout_rate,
        detail_idr_dim=detail_idr_dim,
        detail_feature_channel=detail_feature_channel,
        detail_layer_channels=list(detail_layer_channels),
        detail_num_blocks=detail_num_blocks,
        approx_hidden_size=approx_hidden_size,
        approx_cells=approx_cells,
        refine_hidden_dim=refine_hidden_dim,
        wavelet_name=wavelet_name,
        wavelet_level=wavelet_level,
        loss_a_weight=loss_a_weight,
        loss_a_constant_weight=loss_a_constant_weight,
        loss_a_stop_step=loss_a_stop_step,
        loss_d_weight=loss_d_weight,
        loss_recon_mean_weight=loss_recon_mean_weight,
    )
    return model
