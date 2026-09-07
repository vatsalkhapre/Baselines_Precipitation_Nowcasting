"""EarthFarseer + FALFCL, with the dead parameter blocks removed.

No FACL/FALFCL variant of EarthFarseer existed in ./models. The live
implementation is `models/Earthfarseer/model.py` (NOT the top-level
`models/earthfarseer.py`, which is referenced nowhere in the repo and is the
unadapted, T_in==T_out-locked copy).

The live model already implements the paper's Temporal Projection (Eq. 9) as a
`TemporalProjection` module applied after `self.dec`, and already drops the
`+ skip_feature` residual that otherwise ties output length to input length.
So no horizon work is needed here; verified 5 -> 20 end to end.

DEAD PARAMETERS REMOVED (20,682,721 params, 13.9% of the model):

  * `enc`             7,088,640  -- constructed but never referenced in
                                    forward(). This is dead in the OFFICIAL
                                    EarthFarseer repo too, i.e. it is the
                                    authors' own bug, not one we introduced.
  * `skip_conneciton` 13,594,081 -- live upstream (feeds the `+ skip_feature`
                                    residual), but dead HERE because our
                                    horizon adaptation drops that residual. Our
                                    adaptation orphaned it.

Both were confirmed dead empirically, not by reading: after a full
forward+backward, 0/16 and 0/322 of their parameters had gradients.

Removing them matters beyond the parameter count: the runner constructs DDP
with find_unused_parameters=False, which hard-errors on unused parameters if
this model is ever run on more than one GPU.

  * `Mlp.fc2`      31,469,568  -- allocated and never called. `Mlp.forward`
                                    runs fc1 -> act -> drop -> fc3 -> drop and
                                    returns, skipping fc2 entirely. Also the
                                    authors' bug (byte-identical upstream), in
                                    both copies of the class.

The fc2 case was initially left in place on the reasoning that the authors'
released code is the tiebreaker. That was reversed after measuring it: 31.5M
parameters is 24.6% of the model, and reporting a 127.9M parameter count for a
network that trains 96.4M would misstate the capacity column of the results
table. Removal is provably behaviour-preserving -- fc2 is never called, so
outputs are bit-identical with or without it. The three figures are:

    148,569,327   as constructed upstream
    127,886,606   minus enc + skip_conneciton
     96,417,038   minus fc2 as well  <- trainable capacity, the number to report

Loss: native MSE replaced with FALFCL per the run protocol (EarthFarseer has a
single forecast loss term, so a straight substitution is correct here -- unlike
AlphaPre and PhyDNet, which have auxiliary terms that must stay native).
"""

import torch
import torch.nn as nn

from models.Earthfarseer.model import Earthfarseer_model
from utils.utilspp import RandomScheduling


class EarthFarseerFALFCL(nn.Module):
    def __init__(self, total_steps, T_in, T_out, C, H, W,
                 hid_S=512, hid_T=256, N_S=4, N_T=8,
                 incep_ker=(3, 5, 7, 11), groups=8):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out

        self.model = Earthfarseer_model(
            shape_in=(T_in, C, H, W), hid_S=hid_S, hid_T=hid_T, N_S=N_S,
            N_T=N_T, incep_ker=list(incep_ker), groups=groups, T_out=T_out,
        )

        # --- strip the dead blocks (see module docstring) ---
        for dead in ('enc', 'skip_conneciton'):
            if hasattr(self.model, dead):
                delattr(self.model, dead)
        # Mlp.fc2 is allocated but never called by Mlp.forward (authors' bug,
        # present upstream). Removing it cannot change outputs.
        for mod in self.model.modules():
            if type(mod).__name__ == 'Mlp' and hasattr(mod, 'fc2'):
                delattr(mod, 'fc2')

        self.criterion = RandomScheduling(total_steps, 1, 0.1)
        self.register_buffer('facl_step_buf', torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self.criterion.step = int(self.facl_step_buf.item())

    def forward(self, x):
        return self.model(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        pred = self.forward(frames_in)
        if not compute_loss:
            return pred, None
        loss = self.criterion(pred, frames_gt)
        self.facl_step_buf.fill_(int(self.criterion.step))
        return pred, {'total_loss': loss}


def get_model(total_steps, input_shape=(128, 128), T_in=5, T_out=20,
              img_channels=1, hid_S=512, hid_T=256, N_S=4, N_T=8, **kwargs):
    H, W = input_shape
    return EarthFarseerFALFCL(total_steps, T_in=T_in, T_out=T_out,
                              C=img_channels, H=H, W=W,
                              hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T)
