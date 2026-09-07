"""
Transfer of learned parameters from a THE_GABOR controlled run (donor) into the
full per-subband DAWN-Cast model (recipient), plus optional freezing.

Donor   : THE_GABOR/checkpoints/Gabor_latent_SEVIR_<regime>_seed<N>/checkpoints/
          best_model.pt          (all tensors at the best-validation step)
          gabor_state_best.pt    (same step, Gabor organised per subband)
Recipient: THE_GABOR.models.dawncast_transfer.DAWNCastPerSubbandForecaster

Component name mapping (donor -> recipient), for wavelet_level=J, hf_mode='separate':

    net.block_ll.gabor.*        -> dawncast.wgtm.fat_ll.gabor.*                 ('LL')
    net.blocks_hf.{i}.gabor.*   -> dawncast.wgtm.fat_hf_streams.{i}.gabor.*     ('HF_level_{i+1}')
    net.block_ll.mlp.*          -> dawncast.wgtm.fat_ll.mlp.*
    net.blocks_hf.{i}.mlp.*     -> dawncast.wgtm.fat_hf_streams.{i}.mlp.*
    net.lifting.*               -> dawncast.lifting.*
    net.projection.*            -> dawncast.projection.*

Both models define lifting/projection/GaborLayer/FATBlock from the same code, so
these tensors are shape-identical; every copy is shape-checked anyway.

IMPORTANT -- freq_multiplier is NOT a learnable tensor.  The Gabor response is
    sin(freq_multiplier * freq * linear(x)) * exp(-0.5 * D * gamma)
so transferring `freq` while leaving the recipient's own freq_multiplier in
place would silently rescale the learned function.  `donor_freq_multipliers()`
reads the donor's value per subband so the caller can pass it straight into the
per-subband model and reproduce the donor's function exactly.
"""

import os.path as osp
from collections import OrderedDict

import torch

COMPONENTS = ('gabor', 'mlp', 'lifting', 'projection')


def donor_paths(root, regime, seed=0, space='latent'):
    base = osp.join(root, f'Gabor_{space}_SEVIR_{regime}_seed{seed}', 'checkpoints')
    return {
        'best_model': osp.join(base, 'best_model.pt'),
        'final_model': osp.join(base, 'final_model.pt'),
        'gabor_state_best': osp.join(base, 'gabor_state_best.pt'),
        'gabor_state_final': osp.join(base, 'gabor_state.pt'),
    }


def load_donor(root, regime, seed=0, which='best', space='latent'):
    """Returns (state_dict, gabor_state, meta)."""
    p = donor_paths(root, regime, seed, space)
    mk = 'best_model' if which == 'best' else 'final_model'
    gk = 'gabor_state_best' if which == 'best' else 'gabor_state_final'
    for k in (mk, gk):
        if not osp.exists(p[k]):
            raise FileNotFoundError(f'donor checkpoint missing: {p[k]}')
    ck = torch.load(p[mk], map_location='cpu', weights_only=False)
    gs = torch.load(p[gk], map_location='cpu', weights_only=False)
    if ck['step'] != gs['step']:
        raise ValueError(f"donor mismatch: model step {ck['step']} != "
                         f"gabor_state step {gs['step']}")
    meta = {'regime': regime, 'seed': seed, 'which': which,
            'step': int(ck['step']), 'model_path': p[mk],
            'subbands': list(gs['gabor'])}
    return ck['model'], gs['gabor'], meta


def donor_freq_multipliers(gabor_state, subbands):
    """Per-subband freq_multiplier the donor actually trained with."""
    return [float(gabor_state[s]['freq_multiplier']) for s in subbands]


def _donor_prefixes(subbands):
    """recipient prefix -> donor prefix, per FAT block."""
    out = OrderedDict()
    for i, name in enumerate(subbands):
        if name == 'LL':
            out['dawncast.wgtm.fat_ll'] = 'net.block_ll'
        elif name == 'HF_shared':
            raise ValueError("hf_mode='shared' cannot map a donor with per-level "
                             "HF subbands; use hf_mode='separate'")
        else:
            lvl = int(name.rsplit('_', 1)[1]) - 1
            out[f'dawncast.wgtm.fat_hf_streams.{lvl}'] = f'net.blocks_hf.{lvl}'
    return out


def build_transfer_map(model, donor_sd, subbands, components):
    """
    Returns {recipient_key: donor_tensor} for the requested components.
    Raises on any missing key or shape mismatch -- transfer is never partial
    or silent.
    """
    bad = [c for c in components if c not in COMPONENTS]
    if bad:
        raise ValueError(f'unknown components {bad}, expected subset of {COMPONENTS}')

    recipient_sd = model.state_dict()
    mapping = OrderedDict()
    pairs = []

    fat = _donor_prefixes(subbands)
    for r_pre, d_pre in fat.items():
        for comp in ('gabor', 'mlp'):
            if comp in components:
                pairs.append((f'{r_pre}.{comp}.', f'{d_pre}.{comp}.'))
    if 'lifting' in components:
        pairs.append(('dawncast.lifting.', 'net.lifting.'))
    if 'projection' in components:
        pairs.append(('dawncast.projection.', 'net.projection.'))

    for r_pre, d_pre in pairs:
        r_keys = [k for k in recipient_sd if k.startswith(r_pre)]
        if not r_keys:
            raise KeyError(f'recipient has no tensors under {r_pre}')
        for rk in r_keys:
            dk = d_pre + rk[len(r_pre):]
            if dk not in donor_sd:
                raise KeyError(f'donor missing {dk} (for recipient {rk})')
            if donor_sd[dk].shape != recipient_sd[rk].shape:
                raise ValueError(f'shape mismatch {rk}: recipient '
                                 f'{tuple(recipient_sd[rk].shape)} vs donor '
                                 f'{tuple(donor_sd[dk].shape)}')
            mapping[rk] = donor_sd[dk]
    return mapping


def apply_transfer(model, mapping):
    """Copy donor tensors in place. Returns the number of tensors written."""
    sd = model.state_dict()
    for k, v in mapping.items():
        sd[k].copy_(v)
    return len(mapping)


def verify_transfer(model, mapping):
    """Confirm every transferred tensor now equals its donor value exactly."""
    sd = model.state_dict()
    bad = [k for k, v in mapping.items() if not torch.equal(sd[k].cpu(), v.cpu())]
    return (not bad), bad


def freeze_components(model, subbands, components):
    """
    Set requires_grad=False on the named components.
    Returns (frozen_tensor_names, frozen_param_count).
    """
    prefixes = []
    fat = _donor_prefixes(subbands)
    for r_pre in fat:
        for comp in ('gabor', 'mlp'):
            if comp in components:
                prefixes.append(f'{r_pre}.{comp}.')
    if 'lifting' in components:
        prefixes.append('dawncast.lifting.')
    if 'projection' in components:
        prefixes.append('dawncast.projection.')

    frozen, n = [], 0
    for name, p in model.named_parameters():
        if any(name.startswith(pre) for pre in prefixes):
            p.requires_grad = False
            frozen.append(name)
            n += p.numel()
    return frozen, n


def trainable_report(model):
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fr = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {'trainable': tr, 'frozen': fr, 'total': tr + fr}
