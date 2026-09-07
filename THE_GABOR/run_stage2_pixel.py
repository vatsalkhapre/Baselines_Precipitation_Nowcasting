"""
STAGE 2 (pixel) — full DAWN-Cast, Gabor initialised from the Stage-1 donor.

    python -m THE_GABOR.run_stage2_pixel --dataset cikm \
        --donor_run Stage1_pixel_cikm_seed0 --run_name Stage2_pixel_cikm_seed0 \
        --wave db4 --wavelet_level 2 --afno_blocks 1 --afno_hidden_size_factor 1 \
        --k_spatial 7

Also serves Part B/C ablations (`--ablation <key>`) and item D
(`--transfer` empty -> random Gabor init, everything else identical).

Key correctness point: the donor's `freq_multiplier` (1.0) is carried over, since
the Gabor computes sin(freq_multiplier * freq * linear(x)) and DAWN-Cast's own
default would rescale every transferred frequency.
"""

import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from THE_GABOR.datasets.pixel_datasets import build_pixel_loaders, frames_for
from THE_GABOR.models.dawncast_ablations import get_model as ablation_get_model
from THE_GABOR.models.dawncast_transfer import get_model as dawncast_get_model
from THE_GABOR.models.dawncast_transfer import subband_names
from THE_GABOR.run_stage1_pixel import Stage1PixelExperiment
from THE_GABOR.utils import gabor_transfer as gt
from THE_GABOR.utils.experiment import DEFAULT_OUTPUT_ROOT, base_parser


class Stage2PixelExperiment(Stage1PixelExperiment):
    """Reuses Stage-1's pixel data/validation path; swaps in full DAWN-Cast."""

    space = 'pixel'

    @property
    def model_name(self):
        # The init-checkpoint signature hashes the model name, so each ablation
        # must report a distinct one -- otherwise they all resolve to the same
        # `initial_pixel_<sig>_seed0.pt` and try to load each other's weights
        # (different architectures -> strict load fails).
        ab = getattr(self.args, 'ablation', 'none')
        return ('DAWNCastPerSubband' if ab in (None, '', 'none')
                else f'DAWNCastAblation_{ab}')

    def build_model(self):
        a = self.args
        subs = subband_names(a.wavelet_level, a.hf_mode)
        fm = getattr(self, '_donor_fm', None) or [a.freq_multiplier] * len(subs)
        common = dict(
            T_in=a.frames_in, T_out=a.frames_out, img_channels=a.img_channel,
            dim=a.hidden_dim, afno_blocks=a.afno_blocks,
            sparsity_threshold=a.sparsity_threshold,
            afno_hidden_size_factor=a.afno_hidden_size_factor,
            weight_scale=a.weight_scale, alpha=a.alpha, beta=a.beta,
            size_factor=a.size_factor, total_steps=self.total_steps,
            const_ratio=a.facl_const_ratio, k_spatial=a.k_spatial,
            wave=a.wave, wavelet_level=a.wavelet_level, hf_mode=a.hf_mode)
        if a.ablation and a.ablation != 'none':
            # ablations use scalar freq_multiplier (no donor)
            return ablation_get_model(ablation=a.ablation,
                                      freq_multiplier=a.freq_multiplier, **common)
        return dawncast_get_model(freq_multiplier=fm, **common)

    def wandb_config(self):
        cfg = super().wandb_config()
        a = self.args
        # These define the SRST/STR geometry; without them a checkpoint cannot
        # be rebuilt for evaluation.
        cfg.update({'afno_blocks': a.afno_blocks,
                    'afno_hidden_size_factor': a.afno_hidden_size_factor,
                    'sparsity_threshold': a.sparsity_threshold,
                    'k_spatial': a.k_spatial,
                    'ablation': getattr(a, 'ablation', 'none'),
                    'sevir_regime': getattr(a, 'sevir_regime', 'all')})
        return cfg

    def after_init_load(self):
        a = self.args
        if a.ablation and a.ablation != 'none':
            print(f'[ablation] {a.ablation} — no donor transfer')
            self.cfg.update({'ablation': a.ablation, 'transfer': 'none',
                             'freeze': 'none', 'donor_run': None})
            return
        if not a.transfer:
            print('[stage2] no transfer — random Gabor init (item D control)')
            self.cfg.update({'ablation': 'none', 'transfer': 'none',
                             'freeze': 'none', 'donor_run': None})
            return
        subs = subband_names(a.wavelet_level, a.hf_mode)
        mapping = gt.build_transfer_map(self.model, self._donor_sd, subs, a.transfer)
        n = gt.apply_transfer(self.model, mapping)
        ok, bad = gt.verify_transfer(self.model, mapping)
        if not ok:
            raise RuntimeError(f'transfer verification failed: {bad[:5]}')
        print(f'[stage2] transferred {n} tensors from {a.donor_run} '
              f'(step {self._donor_meta["step"]}), freq_multipliers={self._donor_fm}')
        if a.freeze:
            frozen, nfz = gt.freeze_components(self.model, subs, a.freeze)
            print(f'[stage2] froze {len(frozen)} tensors / {nfz:,} params')
        rep = gt.trainable_report(self.model)
        print(f'[stage2] trainable={rep["trainable"]:,} frozen={rep["frozen"]:,}')
        self.cfg.update({'ablation': 'none', 'donor_run': a.donor_run,
                         'donor_step': self._donor_meta['step'],
                         'transfer': ','.join(a.transfer),
                         'freeze': ','.join(a.freeze) if a.freeze else 'none',
                         'trainable_params': rep['trainable'],
                         'frozen_params': rep['frozen']})


def main():
    p = base_parser('pixel')
    p.add_argument('--preprocessing', type=int, default=0)
    p.add_argument('--sevir_regime', type=str, default='all',
                   choices=['all', 'random', 'storm'],
                   help='Part F: restrict SEVIR to one regime')
    p.add_argument('--donor_run', type=str, default=None,
                   help='Stage-1 run name under --donor_root')
    p.add_argument('--donor_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--donor_which', type=str, default='best', choices=['best', 'final'])
    p.add_argument('--transfer', nargs='*', default=['gabor'])
    p.add_argument('--freeze', nargs='*', default=[])
    p.add_argument('--ablation', type=str, default='none')
    p.add_argument('--afno_blocks', type=int, default=4)
    p.add_argument('--afno_hidden_size_factor', type=int, default=4)
    p.add_argument('--sparsity_threshold', type=float, default=0.01)
    p.add_argument('--k_spatial', type=int, default=3)
    # DAWN-Cast's native low/high Gabor init, for ablations (no donor).
    # Reproduces the original per-level interpolation explicitly:
    #   LL   -> *_low
    #   HF_1 -> *_high
    #   HF_i -> interpolated toward (low+high)/2, as in WGTMBlock
    p.add_argument('--freq_multiplier_low', type=float, default=None)
    p.add_argument('--freq_multiplier_high', type=float, default=None)
    p.add_argument('--weight_scale_low', type=float, default=None)
    p.add_argument('--weight_scale_high', type=float, default=None)
    p.add_argument('--alpha_low', type=float, default=None)
    p.add_argument('--alpha_high', type=float, default=None)
    p.add_argument('--beta_low', type=float, default=None)
    p.add_argument('--beta_high', type=float, default=None)
    args = p.parse_args()

    fi, fo = frames_for(args.dataset)
    args.frames_in, args.frames_out = fi, fo
    args.seq_len = fi + fo
    args.img_channel = 1
    if args.run_name is None:
        args.run_name = f'Stage2_pixel_{args.dataset}_seed{args.seed}'

    def _lowhigh(lo, hi, level):
        """Original WGTMBlock schedule: [LL=lo, HF_1=hi, HF_i interpolated to mid]."""
        out = [lo]
        mid = (lo + hi) / 2.0
        for i in range(level):
            if level == 1:
                out.append(hi)
            else:
                a = i / (level - 1)
                out.append(hi * (1 - a) + mid * a)
        return out

    if args.freq_multiplier_low is not None and args.freq_multiplier_high is not None:
        args.freq_multiplier = _lowhigh(args.freq_multiplier_low,
                                        args.freq_multiplier_high, args.wavelet_level)
        print(f'[init] per-subband freq_multiplier = {args.freq_multiplier}')
    if args.weight_scale_low is not None and args.weight_scale_high is not None:
        args.weight_scale = _lowhigh(args.weight_scale_low,
                                     args.weight_scale_high, args.wavelet_level)
        print(f'[init] per-subband weight_scale    = {args.weight_scale}')
    if args.alpha_low is not None and args.alpha_high is not None:
        args.alpha = _lowhigh(args.alpha_low, args.alpha_high, args.wavelet_level)
        print(f'[init] per-subband alpha           = {args.alpha}')
    if args.beta_low is not None and args.beta_high is not None:
        args.beta = _lowhigh(args.beta_low, args.beta_high, args.wavelet_level)
        print(f'[init] per-subband beta            = {args.beta}')

    cls = Stage2PixelExperiment
    use_donor = (args.ablation in (None, '', 'none')) and bool(args.transfer)
    if use_donor:
        if not args.donor_run:
            raise SystemExit('--donor_run is required unless --ablation is set '
                             'or --transfer is empty')
        base = osp.join(args.donor_root, args.donor_run, 'checkpoints')
        mk = 'best_model.pt' if args.donor_which == 'best' else 'final_model.pt'
        gk = 'gabor_state_best.pt' if args.donor_which == 'best' else 'gabor_state.pt'
        ck = torch.load(osp.join(base, mk), map_location='cpu', weights_only=False)
        gs = torch.load(osp.join(base, gk), map_location='cpu', weights_only=False)
        if ck['step'] != gs['step']:
            raise SystemExit(f"donor mismatch: {ck['step']} vs {gs['step']}")
        subs = subband_names(args.wavelet_level, args.hf_mode)
        missing = [s for s in subs if s not in gs['gabor']]
        if missing:
            raise SystemExit(f'donor lacks subbands {missing}; has {list(gs["gabor"])}')
        cls._donor_sd = ck['model']
        cls._donor_meta = {'step': int(ck['step']), 'run': args.donor_run}
        cls._donor_fm = gt.donor_freq_multipliers(gs['gabor'], subs)
        print(f'[donor] {args.donor_run} step={ck["step"]} '
              f'subbands={list(gs["gabor"])} freq_multipliers={cls._donor_fm}')
    cls(args).train()


if __name__ == '__main__':
    main()
