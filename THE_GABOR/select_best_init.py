"""
Part C gate: pick the better MeteoNet model, then emit the Gabor init the
ablations must use.

Compares the newly trained Part-A Stage-2 run against the PREVIOUS best, whose
published test scores are the DAWN-Cast row of `baseline_work/results_table.tex`
(so the previous checkpoint does not need re-evaluating).

Selection rule from the prompt: CSI-M and/or HSS must improve, and SSIM/PSNR must
not have collapsed. Writes a shell-sourceable file of init flags for whichever
config won, and prints the verdict for RUN_STATE.md.

    python -m THE_GABOR.select_best_init --dataset meteo \
        --run Stage2_pixel_meteo_seed0 --out THE_GABOR/logs/_runlogs/meteo_init.env
"""

import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

# Published DAWN-Cast rows (results_table.tex) = the "previous best" to beat.
BASELINE = {
    'meteo':    dict(csi=0.4529, hss=0.5890, ssim=0.8419, mse=9.31),
    'cikm':     dict(csi=0.3411, hss=0.4385, ssim=0.6087, mse=34.13),
    'shanghai': dict(csi=0.4525, hss=0.5918, ssim=0.7301, mse=25.33),
    'sevir':    dict(csi=0.3787, hss=0.4847, ssim=0.6821, mse=340.90),
}

# Gabor init that produced each previous-best model (from its params.yaml).
PREV_INIT = {
    'meteo': ('--freq_multiplier_low 1.09 --freq_multiplier_high 1.12 '
              '--weight_scale_low 0.1 --weight_scale_high 1.0 '
              '--alpha_low 1.0 --alpha_high 1.0 '
              '--beta_low 0.0995 --beta_high 0.1643'),
}
# Equivalent neutral init when the 2-stage model wins: Stage 1 trained its Gabor
# at freq_multiplier=1.0 with a single weight_scale/alpha/beta, so an ablation
# (which has no donor) reproduces that starting point.
TWOSTAGE_INIT = '--freq_multiplier 1.0 --weight_scale 0.1 --alpha 1.0 --beta 1.0'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--output_root', default=None)
    ap.add_argument('--num_workers', type=int, default=8)
    args = ap.parse_args()

    from THE_GABOR.eval_pixel import evaluate_pixel_run
    base = BASELINE[args.dataset]
    try:
        new = evaluate_pixel_run(args.run, num_workers=args.num_workers,
                                 output_root=args.output_root)
    except Exception as e:                                    # noqa: BLE001
        print(f'[select] evaluation FAILED ({e!r}); falling back to PREVIOUS best init')
        new = None

    if new is None:
        winner, flags = 'previous (eval failed)', PREV_INIT.get(args.dataset, TWOSTAGE_INIT)
    else:
        better = (new['test_csi'] > base['csi']) or (new['test_hss'] > base['hss'])
        collapsed = (new['test_ssim'] < 0.9 * base['ssim'])
        print(f"[select] {args.dataset}: new CSI={new['test_csi']:.4f} HSS={new['test_hss']:.4f} "
              f"SSIM={new['test_ssim']:.4f} | prev CSI={base['csi']} HSS={base['hss']} "
              f"SSIM={base['ssim']}")
        if better and not collapsed:
            winner, flags = '2-stage (new)', TWOSTAGE_INIT
        else:
            winner, flags = 'previous best', PREV_INIT.get(args.dataset, TWOSTAGE_INIT)
        if better and collapsed:
            print('[select] WARNING: CSI/HSS improved but SSIM collapsed >10% — kept previous')

    os.makedirs(osp.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(f'# winner: {winner}\nABL_INIT="{flags}"\n')
    print(f'[select] winner = {winner}')
    print(f'[select] ablation init -> {flags}')
    print(f'[select] wrote {args.out}')


if __name__ == '__main__':
    main()
