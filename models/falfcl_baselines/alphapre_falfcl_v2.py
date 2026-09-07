"""AlphaPre + FALFCL, protocol-correct variant.

Why this file exists
--------------------
`models/alphapre_falfcl.py` (the pre-existing FACL adaptation) replaces the
*entire* AlphaPre objective with FALFCL:

    loss = self.faclloss(pred, frames_gt)

That drops the phase loss, the amplitude loss, the A-Net loss and the
amp_weight decay schedule -- i.e. it turns AlphaPre into a plain
FALFCL-supervised backbone and removes the amplitude/phase disentanglement the
paper is about. The run protocol requires FALFCL on the *base regression term
only*, with the three native auxiliary terms preserved.

This file keeps the architecture byte-identical (it subclasses the original
AlphaPre) and only rewrites `predict()`:

    loss += falfcl(pred, frames_gt)          <- FALFCL substituted HERE, only
    loss += pha_weight  * pha_loss           <- native
    loss += amp_weight  * amp_loss           <- native MSE on FFT magnitudes
    loss += anet_weight * anet_loss          <- native MSE

Applying FALFCL to `amp_loss` would run an FFT-based loss on tensors that are
already FFT magnitudes, which is meaningless.

Resumability
------------
Upstream, `itr` and `amp_weight` are plain Python attributes and the FALFCL
scheduler's `step` is a plain int, so none of them survive a checkpoint. With
preemption a routine event, a resumed run would silently restart both the
amp_weight decay and the FALFCL curriculum at step 0 while the optimizer step
count carried on. All three are registered as buffers here so they round-trip
through `state_dict()`.
"""

import torch
import torch.nn as nn

from models.alphapre import AlphaPre
from utils.utilspp import RandomScheduling


class AlphaPreFALFCL(AlphaPre):
    def __init__(self, total_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FALFCL replaces ONLY the base regression term. self.criterion stays
        # nn.MSELoss() and continues to serve amp_loss and anet_loss.
        self.faclloss = RandomScheduling(total_steps, 1, 0.1)

        # --- persistent step-dependent state (see module docstring) ---
        self.register_buffer('itr_buf', torch.zeros((), dtype=torch.long))
        self.register_buffer('amp_weight_buf', torch.tensor(float(self.amp_weight)))
        self.register_buffer('facl_step_buf', torch.zeros((), dtype=torch.long))

    # -- keep the plain attributes and the buffers in sync ------------------
    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.itr = int(self.itr_buf.item())
        self.amp_weight = float(self.amp_weight_buf.item())
        self.faclloss.step = int(self.facl_step_buf.item())

    def _sync_buffers(self):
        self.itr_buf.fill_(int(self.itr))
        self.amp_weight_buf.fill_(float(self.amp_weight))
        self.facl_step_buf.fill_(int(self.faclloss.step))

    def loss_weights(self):
        """Both scheduled weights, for logging every validation step."""
        prob_idx = min(self.faclloss.step, len(self.faclloss.prob_thres) - 1)
        return {
            'amp_weight': float(self.amp_weight),
            'falfcl_prob_thres': float(self.faclloss.prob_thres[prob_idx].item()),
        }

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        B = frames_in.shape[0]
        xt, xps, xas, x_phas_t, x_amps = self(frames_in, frames_gt, compute_loss)
        pred = xt

        if not compute_loss:
            return pred, None

        # amp_weight decay schedule (native AlphaPre behaviour, restored)
        if self.itr < self.aweight_stop_steps:
            self.amp_weight -= self.sampling_changing_rate
        else:
            self.amp_weight = 0.

        loss = 0.

        # (1) base regression term -- FALFCL substituted here and nowhere else
        base_loss = self.faclloss(pred, frames_gt)
        loss += base_loss

        # (2) phase loss -- native, does not use self.criterion
        frames_fft = torch.fft.rfft2(frames_gt)
        frames_pha = torch.angle(frames_fft)
        frames_abs = torch.abs(frames_fft)
        pha_loss = (1 - torch.cos(frames_pha * self.spec_mask - x_phas_t * self.spec_mask)).sum() \
            / (self.spec_mask.sum() * B * self.aft_seq_length * self.input_dim)
        loss += self.pha_weight * pha_loss

        # (3) amplitude loss -- native MSE on FFT magnitudes
        xas_fft = torch.fft.rfft2(xas)
        xas_abs = torch.abs(xas_fft)
        amp_loss = self.criterion(xas_abs, frames_abs)
        loss += self.amp_weight * amp_loss

        # (4) A-Net loss -- native MSE
        anet_loss = self.criterion(xas, frames_gt)
        loss += self.anet_weight * anet_loss

        self._sync_buffers()

        loss = {
            'total_loss': loss,
            'base_falfcl_loss': base_loss,
            'phase_loss': self.pha_weight * pha_loss,
            'ampli_loss': self.amp_weight * amp_loss,
            'anet_loss': self.anet_weight * anet_loss,
            'amp_weight': float(self.amp_weight),
            'falfcl_prob_thres': self.loss_weights()['falfcl_prob_thres'],
        }
        return pred, loss


def get_model(
    total_steps,
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
    return AlphaPreFALFCL(
        total_steps,
        pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape,
        input_dim=img_channels, hidden_dim=dim, n_layers=n_layers,
        spec_num=spec_num, pha_weight=pha_weight, anet_weight=anet_weight,
        amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
    )
