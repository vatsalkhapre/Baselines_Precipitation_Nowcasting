"""PhyDNet + FALFCL.

No FACL/FALFCL variant of PhyDNet existed in ./models. Note that the runner on
host .205 contains a `phydnet_falfcl` branch importing `models.phydnet_facl`,
but that module exists on no server -- the file used for the completed
PhyDNet-SEVIR run was not kept. This is a new file, not a recovery of that one.

Architecture inherited unchanged from `models/phydnet/phydnet.py` (OpenSTL's
PhyDNet: PhyCell input_dim=64 F_hidden=[49] 7x7, ConvLSTM [128,128,64] 3x3,
K2M([7,7]) moment constraint, patch_size=4).

LOSS -- this is the part that needs care. PhyDNet uses `self.criterion` in
THREE places:

  (1) encoder phase: next-frame reconstruction over the INPUT sequence
  (2) decoder phase: forecast vs target            <- the actual prediction term
  (3) the K2M moment constraint: a 49x7x7 kernel-moment tensor vs `constraints`

Only (2) is a spatial forecast. Substituting FALFCL globally would run an
FFT-based image loss over a 7x7 moment matrix in (3), which is the same failure
mode the protocol warns about for AlphaPre's amp_loss -- meaningless, and (3)
is not even 5D so FALFCL's `_, _, _, H, W = pred.shape` would raise.

So: FALFCL replaces (2) only. (1) and (3) stay native MSE.
"""

import torch
import torch.nn as nn

from models.phydnet.phydnet import PhyDNet_Model
from utils.utilspp import RandomScheduling


class PhyDNetFALFCL(PhyDNet_Model):
    def __init__(self, total_steps, in_shape, T_in, T_out, device):
        super().__init__(in_shape=in_shape, T_in=T_in, T_out=T_out, device=device)
        # self.criterion stays nn.MSELoss() and continues to serve (1) and (3).
        self.faclloss = RandomScheduling(total_steps, 1, 0.1)
        self.register_buffer('facl_step_buf', torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.faclloss.step = int(self.facl_step_buf.item())

    def forward(self, input_tensor, target_tensor, constraints, teacher_forcing_ratio=0.0):
        loss = 0.
        preds = []

        # (1) encoder-phase reconstruction over the input sequence -- native MSE
        for ei in range(self.pre_seq_length - 1):
            _, _, output_image, _, _ = self.encoder(input_tensor[:, ei, :, :, :], (ei == 0))
            loss += self.criterion(output_image, input_tensor[:, ei + 1, :, :, :])

        # (2) decoder-phase forecast -- FALFCL, applied once over the full
        #     (B, T_out, C, H, W) rollout rather than per frame, because FALFCL
        #     is defined on a 5D sequence tensor.
        decoder_input = input_tensor[:, -1, :, :, :]
        for di in range(self.aft_seq_length):
            _, _, output_image, _, _ = self.encoder(decoder_input)
            preds.append(output_image)
            decoder_input = output_image          # teacher forcing off, as upstream
        pred_seq = torch.stack(preds, dim=1)
        forecast_loss = self.faclloss(pred_seq, target_tensor)
        loss += forecast_loss

        # (3) K2M moment constraint -- native MSE, never FALFCL
        for b in range(0, self.encoder.phycell.cell_list[0].input_dim):
            filters = self.encoder.phycell.cell_list[0].F.conv1.weight[:, b, :, :]
            m = self.k2m(filters.double()).float()
            loss += self.criterion(m, constraints)

        self.facl_step_buf.fill_(int(self.faclloss.step))
        return loss, pred_seq

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        constraints = self._get_constraints()
        if compute_loss:
            loss, pred = self(frames_in, frames_gt, constraints)
            return pred, {'total_loss': loss}
        return self.inference(frames_in), None


def get_model(total_steps, in_shape, T_in, T_out, device, **kwargs):
    return PhyDNetFALFCL(total_steps, in_shape=in_shape, T_in=T_in, T_out=T_out, device=device)
