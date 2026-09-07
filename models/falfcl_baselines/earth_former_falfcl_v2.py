"""EarthFormer + FALFCL with the FALFCL curriculum step persisted.

models/earth_former_falfcl.py is CORRECT on the loss protocol and is left
untouched. Same single defect as trajGRU_falfcl.py: RandomScheduling.step is a
plain int and does not survive a checkpoint, so a resumed run restarts the
FALFCL curriculum at 0. Persisted here as a buffer.
"""
import torch
from models.earth_former_falfcl import EarthFormer_xy as _Base


class EarthFormerFALFCLv2(_Base):
    def __init__(self, total_steps, in_len, out_len, height, width, **kw):
        super().__init__(total_steps, in_len, out_len, height, width, **kw)
        self.register_buffer('facl_step_buf', torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(self, *a, **k):
        super()._load_from_state_dict(*a, **k)
        self.criterion.step = int(self.facl_step_buf.item())

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        out = super().predict(frames_in, frames_gt, compute_loss)
        self.facl_step_buf.fill_(int(self.criterion.step))
        return out
