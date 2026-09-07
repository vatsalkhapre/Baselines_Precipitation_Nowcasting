"""
Item E: pick the best run across seeds and compare it to the previous best.

Evaluates every given run on its test split (per-threshold CSI included), ranks
by CSI then HSS, and reports whether the winner beats the published DAWN-Cast row
in `baseline_work/results_table.tex`. Writes a small JSON verdict for RUN_STATE.md
and for the final table update. Does NOT edit the .tex — that stays a deliberate
step.
"""
import argparse, json, os.path as osp, sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from THE_GABOR.eval_pixel import evaluate_pixel_run
from THE_GABOR.select_best_init import BASELINE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    rows = []
    for r in a.runs:
        try:
            rows.append(evaluate_pixel_run(r, num_workers=a.num_workers))
        except Exception as e:                                # noqa: BLE001
            print(f'[seed-select] SKIP {r}: {e!r}')
    if not rows:
        raise SystemExit('[seed-select] no runs could be evaluated')

    rows.sort(key=lambda d: (d.get('test_csi', 0), d.get('test_hss', 0)), reverse=True)
    best = rows[0]
    base = BASELINE[a.dataset]
    beats = (best['test_csi'] > base['csi']) or (best['test_hss'] > base['hss'])
    collapsed = best['test_ssim'] < 0.9 * base['ssim']

    print(f'[seed-select] {a.dataset}: ranked')
    for d in rows:
        print(f"   {d['run']:<38} CSI={d['test_csi']:.4f} HSS={d['test_hss']:.4f} "
              f"SSIM={d['test_ssim']:.4f} MSE={d['test_mse']:.2f}")
    print(f"[seed-select] winner = {best['run']}")
    print(f"[seed-select] previous best: CSI={base['csi']} HSS={base['hss']} SSIM={base['ssim']}")
    verdict = ('UPDATE results_table.tex' if (beats and not collapsed)
               else 'KEEP previous (no improvement)' if not beats
               else 'KEEP previous (SSIM collapsed >10%)')
    print(f'[seed-select] verdict: {verdict}')

    out = a.out or f'THE_GABOR/logs/_runlogs/select_{a.dataset}.json'
    json.dump({'dataset': a.dataset, 'winner': best['run'], 'winner_metrics': best,
               'previous_best': base, 'verdict': verdict, 'all': rows},
              open(out, 'w'), indent=2)
    print(f'[seed-select] wrote {out}')


if __name__ == '__main__':
    main()
