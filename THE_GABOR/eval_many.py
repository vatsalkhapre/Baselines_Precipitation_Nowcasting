"""
Evaluate a list of pixel runs on their test split and dump one JSON per run.

    python -m THE_GABOR.eval_many --runs A B C --out_dir ICLR26/eval

Results carry per-threshold CSI (csi_t*), so the two highest-intensity CSI
columns of results_table.tex can be filled for every dataset. Already-evaluated
runs are skipped, so this is safe to re-run.
"""
import argparse, json, os, os.path as osp, sys
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
from THE_GABOR.eval_pixel import evaluate_pixel_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--which', default='best')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for r in a.runs:
        out = osp.join(a.out_dir, f'{r}.json')
        if osp.exists(out) and not a.force:
            print(f'[eval-many] SKIP {r} (already evaluated)'); continue
        try:
            res = evaluate_pixel_run(r, which=a.which, num_workers=a.num_workers)
            json.dump(res, open(out, 'w'), indent=2)
            print(f"[eval-many] OK {r}: CSI={res['test_csi']:.4f} HSS={res['test_hss']:.4f}")
        except Exception as e:                                  # noqa: BLE001
            print(f'[eval-many] FAIL {r}: {e!r}')
    print('[eval-many] done')


if __name__ == '__main__':
    main()
