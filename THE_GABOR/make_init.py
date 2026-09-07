"""
Create the ONE shared initial checkpoint for a (space, seed) pair.

    python -m THE_GABOR.make_init --space pixel  --seed 0
    python -m THE_GABOR.make_init --space latent --seed 0

Both the RANDOM run and the STORM run then load this exact file, so their
initialisation is identical by construction rather than by assumption.
Run this before the training runs; the runners will also create it on demand
if it is missing (and print a notice when they do).
"""

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from THE_GABOR.models.gabor_mlp_model import get_model
from THE_GABOR.utils.experiment import DEFAULT_INIT_ROOT
from THE_GABOR.utils.init_checkpoint import (architecture_signature,
                                             create_initial_checkpoint,
                                             initial_checkpoint_path)


def build_cfg(args):
    return {
        'model': 'GaborMLPControlled',
        'space': args.space,
        'img_channel': args.img_channel,
        'frames_in': args.frames_in,
        'frames_out': args.frames_out,
        'hidden_dim': args.hidden_dim,
        'wave': args.wave,
        'wavelet_level': args.wavelet_level,
        'hf_mode': args.hf_mode,
        'freq_multiplier': args.freq_multiplier,
        'weight_scale': args.weight_scale,
        'alpha': args.alpha,
        'beta': args.beta,
        'size_factor': args.size_factor,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--space', type=str, required=True, choices=['pixel', 'latent'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--init_root', type=str, default=DEFAULT_INIT_ROOT)
    p.add_argument('--img_channel', type=int, default=None)
    p.add_argument('--frames_in', type=int, default=5)
    p.add_argument('--frames_out', type=int, default=None)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--wave', type=str, default='db6')
    p.add_argument('--wavelet_level', type=int, default=2)
    p.add_argument('--hf_mode', type=str, default='separate')
    p.add_argument('--freq_multiplier', type=float, default=1.0)
    p.add_argument('--weight_scale', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--size_factor', type=float, default=1.0)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    if args.img_channel is None:
        args.img_channel = 1 if args.space == 'pixel' else 4
    if args.frames_out is None:
        args.frames_out = 20

    cfg = build_cfg(args)
    sig = architecture_signature(cfg)
    path = initial_checkpoint_path(args.init_root, args.space, args.seed, sig)

    def build():
        return get_model(
            T_in=args.frames_in, T_out=args.frames_out,
            img_channels=args.img_channel, dim=args.hidden_dim,
            weight_scale=args.weight_scale, alpha=args.alpha, beta=args.beta,
            freq_multiplier=args.freq_multiplier, size_factor=args.size_factor,
            wave=args.wave, wavelet_level=args.wavelet_level,
            hf_mode=args.hf_mode, total_steps=1, const_ratio=0.1)

    path, sha = create_initial_checkpoint(build, path, args.seed, cfg,
                                          overwrite=args.overwrite)
    print(f'[make_init] space={args.space} seed={args.seed} signature={sig}')
    print(f'[make_init] path   = {path}')
    print(f'[make_init] sha256 = {sha}')


if __name__ == '__main__':
    main()
