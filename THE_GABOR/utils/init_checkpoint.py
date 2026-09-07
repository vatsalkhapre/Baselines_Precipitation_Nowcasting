"""
Identical-initialisation guarantee (Experiment 1).

Requirement: for every seed, the RANDOM and the STORM model must start from the
EXACT SAME initial checkpoint -- not merely from two models seeded identically.

Procedure implemented here:
    1. ONE model is initialised (under `seed`) by `create_initial_checkpoint`.
    2. Its state_dict is written to  <root>/initial_seed<seed>.pt
    3. Both the RANDOM run and the STORM run call `load_initial_checkpoint`,
       which loads that same file with strict=True.
    4. The sha256 of the file is recorded in every run (printed, written to the
       run directory and logged to W&B) so that "byte-for-byte identical" is
       verifiable after the fact from the two runs alone.

The initial checkpoint is keyed by (space, seed, architecture signature), so a
configuration change cannot silently reuse an incompatible checkpoint.
"""

import hashlib
import json
import os

import numpy as np
import torch


def architecture_signature(cfg):
    """Stable short hash of the architecture-defining configuration."""
    keys = ('model', 'space', 'img_channel', 'frames_in', 'frames_out',
            'hidden_dim', 'wave', 'wavelet_level', 'hf_mode', 'freq_multiplier',
            'weight_scale', 'alpha', 'beta', 'size_factor')
    payload = json.dumps({k: cfg[k] for k in keys}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def initial_checkpoint_path(root, space, seed, signature):
    return os.path.join(root, f'initial_{space}_{signature}_seed{seed}.pt')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_initial_checkpoint(build_model_fn, path, seed, cfg, overwrite=False):
    """
    Initialise ONE model under `seed` and save its initial state_dict.
    Returns (path, sha256).  Existing files are reused unless `overwrite`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and not overwrite:
        return path, sha256_file(path)

    seed_everything(seed)
    model = build_model_fn()
    payload = {
        'model': {k: v.cpu() for k, v in model.state_dict().items()},
        'seed': int(seed),
        'signature': architecture_signature(cfg),
        'config': {k: cfg[k] for k in sorted(cfg)},
    }
    torch.save(payload, path)
    del model
    return path, sha256_file(path)


def load_initial_checkpoint(model, path, expected_signature=None):
    """Load the shared initial checkpoint into `model` (strict)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Initial checkpoint {path} not found. Create it first with "
            f"THE_GABOR/make_init.py so that RANDOM and STORM provably share it."
        )
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if expected_signature is not None and payload.get('signature') != expected_signature:
        raise ValueError(
            f"Initial checkpoint signature mismatch: file has "
            f"{payload.get('signature')}, run expects {expected_signature}."
        )
    missing, unexpected = model.load_state_dict(payload['model'], strict=False)
    # RandomScheduling holds no parameters/buffers, so both lists must be empty.
    if missing or unexpected:
        raise RuntimeError(
            f"Initial checkpoint does not match model. missing={missing} "
            f"unexpected={unexpected}")
    return payload
