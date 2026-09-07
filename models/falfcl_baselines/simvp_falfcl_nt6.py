"""SimVP+FALFCL as it existed when the completed SimVP runs were trained (N_T=6).

models/simvp_falfcl/simvp_iter.py currently declares N_T=4 in its module-level
`config_dict`. The completed simvp_on_meteo / simvp_on_sevir checkpoints were
trained with SIX MidIncepNet blocks (their state dicts contain hid.enc.4/5 and
hid.dec.4/5), so they cannot be loaded by the current file -- exactly the same
class of drift as WADEPre's refine_hidden_dim: the model file changed after the
runs were produced.

N_T=6 is also what the sibling models/simvp/simvp_iter.py still declares, which
is consistent with 6 having been the original value.

Verified empirically rather than assumed: with N_T=6 both checkpoints strict-load
with 0 missing / 0 unexpected keys and yield 11,063,105 parameters, matching the
"11.06M" recorded in those runs' own logs. With N_T=4 the load fails outright.

This file only overrides the module-level config; the architecture code and the
original file are untouched, so current/future SimVP runs are unaffected.
"""
import models.simvp_falfcl.simvp_iter as _sv

_sv.configs.N_T = 6          # restore the value these checkpoints were trained with


def get_model(total_steps, in_shape, T_in, T_out, **kwargs):
    return _sv.get_model(total_steps=total_steps, in_shape=in_shape,
                         T_in=T_in, T_out=T_out, **kwargs)
