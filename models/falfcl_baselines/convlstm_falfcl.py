"""ConvLSTM (Shi et al. 2015) + FALFCL.

No FACL/FALFCL variant of ConvLSTM existed in ./models -- this is a new file.
The architecture is inherited unchanged from `models/convlstm.py` (2-layer
encoder-forecaster, 64 hidden units, 3x3 kernels, per the paper's Sec 4.2
radar-echo config).

Loss: the paper trains on per-pixel cross-entropy. Per the run protocol,
ConvLSTM is trained with FALFCL instead. FALFCL is a Fourier-domain loss and
must see pixel-space predictions, so it is applied to the post-sigmoid
predictions, not to the raw logits the BCE used.

Resumability: RandomScheduling's `step` is a plain int upstream, so a resumed
run would restart the FALFCL curriculum at 0. Persisted as a buffer here.
"""

import torch
import torch.nn as nn

from models.convlstm import EncoderForecaster
from utils.utilspp import RandomScheduling


class ConvLSTMFALFCL(nn.Module):
    def __init__(self, total_steps, frames_in, frames_out, input_channels=1,
                 hidden_dims=(64, 64), kernel_size=(3, 3)):
        super().__init__()
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.net = EncoderForecaster(
            input_dim=input_channels, hidden_dims=list(hidden_dims),
            kernel_size=kernel_size, num_layers=len(hidden_dims),
        )
        self.criterion = RandomScheduling(total_steps, 1, 0.1)
        self.register_buffer('facl_step_buf', torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.criterion.step = int(self.facl_step_buf.item())

    def forward(self, x):
        return self.net(x, pred_len=self.frames_out)

    def predict(self, frames_in, frames_gt=None, compute_loss=True):
        logits = self.forward(frames_in)
        preds = torch.sigmoid(logits)
        if not compute_loss or frames_gt is None:
            return preds, None
        if frames_gt.shape != preds.shape:
            raise ValueError(f"frames_gt shape {frames_gt.shape} != preds shape {preds.shape}")
        loss = self.criterion(preds, frames_gt)
        self.facl_step_buf.fill_(int(self.criterion.step))
        return preds, {'total_loss': loss}


def get_model(total_steps, T_in=5, T_out=20, img_channels=1,
              hidden_dims=(64, 64), kernel_size=(3, 3), **kwargs):
    return ConvLSTMFALFCL(total_steps, frames_in=T_in, frames_out=T_out,
                          input_channels=img_channels, hidden_dims=hidden_dims,
                          kernel_size=kernel_size)
