"""
Experiment 2 -- full DAWN-Cast on SEVIR STORM latent, initialised from a
THE_GABOR controlled run (donor).

    # 1: Gabor from the matched regime (storm -> storm)
    python -m THE_GABOR.run_dawncast_transfer --donor_regime storm \
        --transfer gabor --run_name DAWNCast_latent_storm_gaborinit_storm_seed0

    # 2: Gabor from the mismatched regime (random -> storm)
    python -m THE_GABOR.run_dawncast_transfer --donor_regime random \
        --transfer gabor --run_name DAWNCast_latent_storm_gaborinit_random_seed0

    # 3: Gabor+MLP+lifting+projection from storm, all frozen
    python -m THE_GABOR.run_dawncast_transfer --donor_regime storm \
        --transfer gabor mlp lifting projection \
        --freeze   gabor mlp lifting projection \
        --run_name DAWNCast_latent_storm_frozen_storm_seed0

Target data is SEVIR STORM latent in every case; only the initialisation
differs.  All runs share one DAWN-Cast initial checkpoint, so the transferred
subset is the only difference between them.

Training objective is FACL only, as in DAWNCastForecaster.
"""

import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch

from THE_GABOR.models.dawncast_transfer import get_model, subband_names
from THE_GABOR.run_latent import LatentGaborExperiment
from THE_GABOR.utils import gabor_transfer as gt
from THE_GABOR.utils.experiment import DEFAULT_AE_CKPT, DEFAULT_OUTPUT_ROOT, base_parser


class DAWNCastTransferExperiment(LatentGaborExperiment):
    space = 'latent'
    model_name = 'DAWNCastPerSubband'   # keeps its init checkpoint distinct

    # ---- recipient model: full DAWN-Cast with per-subband Gabor params ----
    def build_model(self):
        a = self.args
        subs = subband_names(a.wavelet_level, a.hf_mode)
        # Reproduce the donor's freq_multiplier per subband so the transferred
        # `freq` is not silently rescaled (see gabor_transfer docstring).
        fm = getattr(self, '_donor_fm', None) or [a.freq_multiplier] * len(subs)
        return get_model(
            T_in=a.frames_in, T_out=a.frames_out, img_channels=a.img_channel,
            dim=a.hidden_dim, afno_blocks=a.afno_blocks,
            sparsity_threshold=a.sparsity_threshold,
            afno_hidden_size_factor=a.afno_hidden_size_factor,
            weight_scale=a.weight_scale, alpha=a.alpha, beta=a.beta,
            freq_multiplier=fm, size_factor=a.size_factor,
            total_steps=self.total_steps, const_ratio=a.facl_const_ratio,
            k_spatial=a.k_spatial, wave=a.wave,
            wavelet_level=a.wavelet_level, hf_mode=a.hf_mode)

    # ---- donor transfer + freeze, after the shared init, before the optimizer ----
    def after_init_load(self):
        a = self.args
        subs = subband_names(a.wavelet_level, a.hf_mode)
        print('=' * 68)
        print(f'TRANSFER  donor_regime={a.donor_regime}  which={a.donor_which}  '
              f'components={a.transfer}')

        if not a.transfer:
            print('  nothing to transfer (baseline: DAWN-Cast default init)')
        else:
            mapping = gt.build_transfer_map(self.model, self._donor_sd, subs, a.transfer)
            n = gt.apply_transfer(self.model, mapping)
            ok, bad = gt.verify_transfer(self.model, mapping)
            if not ok:
                raise RuntimeError(f'transfer verification failed for {bad[:5]}')
            print(f'  transferred {n} tensors, all verified equal to donor')
            per = {}
            for k in mapping:
                per[k.split('.')[-3] if '.gabor.' in k or '.mlp.' in k
                    else k.split('.')[1]] = per.get(k.split('.')[-3] if '.gabor.' in k
                                                    or '.mlp.' in k else k.split('.')[1], 0) + 1
            print(f'  donor step={self._donor_meta["step"]}  '
                  f'freq_multipliers={self._donor_fm}')

        if a.freeze:
            frozen, nfz = gt.freeze_components(self.model, subs, a.freeze)
            print(f'FREEZE    components={a.freeze} -> {len(frozen)} tensors, '
                  f'{nfz:,} parameters frozen')
        rep = gt.trainable_report(self.model)
        print(f'  trainable={rep["trainable"]:,}  frozen={rep["frozen"]:,}  '
              f'total={rep["total"]:,}')
        print('=' * 68)
        self.cfg.update({
            'donor_regime': a.donor_regime, 'donor_which': a.donor_which,
            'donor_step': self._donor_meta['step'],
            'transfer': ','.join(a.transfer) if a.transfer else 'none',
            'freeze': ','.join(a.freeze) if a.freeze else 'none',
            'trainable_params': rep['trainable'], 'frozen_params': rep['frozen'],
        })

    # ---- Gabor logging still applies: DAWN-Cast exposes gabor_layers() ----


def main():
    p = base_parser('latent')
    p.add_argument('--ae_ckpt_path', type=str, default=DEFAULT_AE_CKPT)
    # donor
    p.add_argument('--donor_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--donor_regime', type=str, default='storm',
                   choices=['random', 'storm'])
    p.add_argument('--donor_seed', type=int, default=0)
    p.add_argument('--donor_which', type=str, default='best', choices=['best', 'final'])
    p.add_argument('--transfer', nargs='*', default=['gabor'],
                   help='subset of: gabor mlp lifting projection (empty = none)')
    p.add_argument('--freeze', nargs='*', default=[],
                   help='subset of: gabor mlp lifting projection')
    # DAWN-Cast specific
    p.add_argument('--afno_blocks', type=int, default=4)
    p.add_argument('--afno_hidden_size_factor', type=int, default=4)
    p.add_argument('--sparsity_threshold', type=float, default=0.01)
    p.add_argument('--k_spatial', type=int, default=3)
    p.add_argument('--target_regime', type=str, default='storm',
                   choices=['random', 'storm', 'all'],
                   help='which SEVIR latent regime the DAWN-Cast model is TRAINED on')
    args = p.parse_args()

    # latent protocol, matching the pixel experiment
    args.img_size, args.img_channel = 32, 4
    args.frames_in, args.frames_out, args.seq_len = 5, 20, 25
    args.regime = args.target_regime

    # Load the donor first: its freq_multipliers define the recipient's Gabor
    # scaling, so they must be known before the model is built.
    exp_cls = DAWNCastTransferExperiment
    donor_sd, gabor_state, meta = gt.load_donor(
        args.donor_root, args.donor_regime, args.donor_seed, args.donor_which)
    subs = subband_names(args.wavelet_level, args.hf_mode)
    missing = [s for s in subs if s not in gabor_state]
    if missing:
        raise ValueError(f'donor lacks subbands {missing}; donor has {list(gabor_state)}')

    exp_cls._donor_sd = donor_sd
    exp_cls._donor_meta = meta
    exp_cls._donor_fm = gt.donor_freq_multipliers(gabor_state, subs)
    print(f'[donor] {meta["model_path"]}')
    print(f'[donor] regime={meta["regime"]} which={meta["which"]} step={meta["step"]} '
          f'subbands={meta["subbands"]} freq_multipliers={exp_cls._donor_fm}')

    exp = exp_cls(args)
    exp.train()


if __name__ == '__main__':
    main()
