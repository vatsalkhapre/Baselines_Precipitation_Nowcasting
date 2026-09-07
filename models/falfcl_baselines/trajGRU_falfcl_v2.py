"""TrajGRU + FALFCL with the FALFCL curriculum step persisted.

models/trajGRU_falfcl.py is CORRECT on the loss protocol (TrajGRU has a single
MSE forecast term, so swapping it wholesale for RandomScheduling is the right
substitution) and is left untouched.

The only thing wrong with it is resumability: RandomScheduling.step is a plain
Python int, so it never enters state_dict(). Under preemption -- a routine event
in this schedule -- a resumed run silently restarts the FALFCL curriculum at
step 0 while the optimizer step count carries on, so the FAL/FCL mixing
probability desyncs from actual training progress. This subclass persists it.

Architecture and loss are otherwise identical to trajGRU_falfcl.py.
"""
import torch
from models.trajGRU_falfcl import TrajGRU_model as _Base


class TrajGRUFALFCLv2(_Base):
    def __init__(self, total_steps, future_seq_len, batch_size, **kw):
        super().__init__(total_steps, future_seq_len, batch_size, **kw)
        self.register_buffer('facl_step_buf', torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(self, *a, **k):
        super()._load_from_state_dict(*a, **k)
        self.criterion.step = int(self.facl_step_buf.item())

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        out = super().predict(frames_in, frames_gt, compute_loss)
        self.facl_step_buf.fill_(int(self.criterion.step))
        return out


def get_model(total_steps, T_out=20, batch_size=4, **kw):
    return TrajGRUFALFCLv2(total_steps, future_seq_len=T_out, batch_size=batch_size)
