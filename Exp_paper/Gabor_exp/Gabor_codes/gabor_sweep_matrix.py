#!/usr/bin/env python3
"""
gabor_sweep_matrix.py
──────────────────────
Build the (freq_low x freq_high) score matrices for the Gabor regime sweep,
one heatmap per metric (CSI-M, CSI-35, CSI-40, HSS, MSE, SSIM, PSNR, ...).

Axes
────
  x-axis = freq_multiplier_low   (LL / large-scale convective bulk)   low -> high sinusoid
  y-axis = freq_multiplier_high  (HF / turbulence)                    low -> high sinusoid

Two log layouts are both supported automatically
─────────────────────────────────────────────────
  * CIKM style: one log.log PER run, each inside its own exp directory
      Exps/<exp_dir>/<backbone>_<dataset>_<exp_note>/logs/log.log
    -> pass --logs_glob 'Exps/multiseed_cikm/*/logs/log.log'
       (or --logs_root and let the script find them)

  * Shanghai style: ALL runs concatenated in ONE log.log (exp_note constant,
    files overwritten but the file handler appends run after run):
      train-block, eval-block, train-block, eval-block, ...
    -> pass --logs_glob '/path/to/shanghai/logs/log.log'  (single file, many runs)

The parser does not care which layout it gets. It scans every file, splits it
into runs, and pairs each run's (freq_low, freq_high) with that run's TEST
metrics. De-duplication keeps the LAST occurrence of any (freq_low, freq_high)
pair (so a re-run overrides an earlier one).

How a run's metrics are located
────────────────────────────────
  * freq_low / freq_high: parsed from the
        "model parameters : {... 'freq_multiplier_low': X, 'freq_multiplier_high': Y ...}"
    line (a python-dict literal, parsed with ast — robust to key order).
  * CSI-M, HSS, MSE, SSIM, PSNR, etc.: from the single-line
        "Test Results: {'csi': ..., 'hss': ..., 'mse': ..., 'ssim': ..., 'psnr': ...}"
    dict. This line marks the END of a run's real test evaluation.
  * CSI-35 / CSI-40 (per-threshold means): NOT in Test Results. They live in the
    evaluation block that immediately PRECEDES the Test Results line, under
        "====================Threshold: 35 with melthod 1===================="
        "<CSI> : 0.229...; [ ... ]"
    We take the scalar before the ';' as the threshold-mean CSI.

A "run" for parsing = the text span from one "model parameters :" line up to
(and including) the next "Test Results:" line. Only spans that contain BOTH a
model-parameters line and a Test Results line are treated as complete test runs;
train-only spans (Valid Results, no Test Results) are skipped.

Usage
─────
  # CIKM (per-run directories)
  python gabor_sweep_matrix.py \
      --logs_glob 'Exps/multiseed_cikm/*/logs/log.log' \
      --dataset cikm --thresholds 35 40 \
      --out_dir cikm_matrices

  # Shanghai (single concatenated log)
  python gabor_sweep_matrix.py \
      --logs_glob '/home/vatsal/.../gabor_exp_shanghai/.../logs/log.log' \
      --dataset shanghai --thresholds 35 40 \
      --out_dir shanghai_matrices

Requirements: numpy, matplotlib  (pandas optional, only for nicer CSV)
"""

import argparse
import ast
import glob
import re
from pathlib import Path

import numpy as np


# ════════════════════════════════════════════════════════════
# Regexes / anchors
# ════════════════════════════════════════════════════════════

RE_MODEL_PARAMS = re.compile(r"model parameters\s*:\s*(\{.*\})\s*$")
RE_TEST_RESULTS = re.compile(r"Test Results\s*:\s*(\{.*\})\s*$")
RE_THRESHOLD_HDR = re.compile(r"={2,}\s*Threshold:\s*(\d+)\s*with\s*melthod\s*\d+\s*={2,}")
# <CSI> : 0.229...;  [ ... ]   -> capture the scalar before ';'
RE_CSI_SCALAR = re.compile(r"<CSI>\s*:\s*([0-9.eE+-]+)\s*;")


# Metrics we pull straight from the Test Results dict, mapped to friendly names.
TEST_DICT_METRICS = {
    'csi':   'CSI-M',
    'csi4':  'CSI-pool-4x4',
    'csi16': 'CSI-pool-16x16',
    'hss':   'HSS',
    'mse':   'MSE',
    'mae':   'MAE',
    'rmse':  'RMSE',
    'psnr':  'PSNR',
    'ssim':  'SSIM',
    'crps':  'CRPS',
    'lpips': 'LPIPS',
}

# Higher-is-better? (controls colormap direction only)
HIGHER_IS_BETTER = {
    'CSI-M': True, 'CSI-pool-4x4': True, 'CSI-pool-16x16': True, 'HSS': True,
    'SSIM': True, 'PSNR': True,
    'MSE': False, 'MAE': False, 'RMSE': False, 'CRPS': False, 'LPIPS': False,
}
# per-threshold CSI (e.g. CSI-35) added dynamically, always higher-is-better.


# ════════════════════════════════════════════════════════════
# Parsing
# ════════════════════════════════════════════════════════════

def _round_key(x, ndigits=2):
    """Round a freq value for use as a dict key so 714.49 and 714.490001 match."""
    return round(float(x), ndigits)


def parse_log_file(path: str, thresholds: list) -> list:
    """
    Parse ONE log file (which may contain one run OR many concatenated runs).

    Returns a list of dicts, one per complete test run:
        {'freq_low': float, 'freq_high': float,
         'CSI-M': float, 'HSS': float, ..., 'CSI-35': float, 'CSI-40': float,
         'source_file': str}
    """
    with open(path, 'r', errors='replace') as f:
        lines = f.readlines()

    runs = []
    # State for the run currently being assembled.
    cur_freqs = None            # (freq_low, freq_high) from the most recent model-params line
    cur_thr_csi = {}            # {threshold_int: csi_scalar} accumulated in the current eval block
    pending_threshold = None    # threshold whose <CSI> line we're waiting for

    for line in lines:
        # --- model parameters line: starts a new run's parameter context ---
        m = RE_MODEL_PARAMS.search(line)
        if m:
            # A new model-parameters line means a new phase (train or eval).
            # Reset per-block threshold accumulation but remember freqs.
            try:
                d = ast.literal_eval(m.group(1))
                cur_freqs = (float(d['freq_multiplier_low']),
                             float(d['freq_multiplier_high']))
            except (ValueError, SyntaxError, KeyError):
                cur_freqs = None
            cur_thr_csi = {}
            pending_threshold = None
            continue

        # --- threshold header: next <CSI> line belongs to this threshold ---
        m = RE_THRESHOLD_HDR.search(line)
        if m:
            pending_threshold = int(m.group(1))
            continue

        # --- <CSI> scalar under a threshold header ---
        if pending_threshold is not None:
            m = RE_CSI_SCALAR.search(line)
            if m:
                cur_thr_csi[pending_threshold] = float(m.group(1))
                pending_threshold = None
                continue

        # --- Test Results line: closes the current run ---
        m = RE_TEST_RESULTS.search(line)
        if m:
            if cur_freqs is None:
                # Test Results with no preceding model-params freqs -> skip.
                cur_thr_csi = {}
                continue
            try:
                td = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                cur_thr_csi = {}
                continue

            run = {'freq_low': cur_freqs[0], 'freq_high': cur_freqs[1],
                   'source_file': path}
            for k, friendly in TEST_DICT_METRICS.items():
                if k in td:
                    run[friendly] = float(td[k])
            # per-threshold CSI from the eval block just parsed
            for thr in thresholds:
                run[f'CSI-{thr}'] = cur_thr_csi.get(thr, np.nan)

            runs.append(run)
            # reset for any subsequent concatenated run
            cur_thr_csi = {}
            # keep cur_freqs? No — next run will set its own via model params.
            cur_freqs = None

    return runs


def collect_runs(log_paths: list, thresholds: list) -> list:
    all_runs = []
    for p in log_paths:
        runs = parse_log_file(p, thresholds)
        all_runs.extend(runs)
        print(f"  parsed {len(runs):>3} test run(s) from {p}")
    return all_runs


def deduplicate(runs: list) -> list:
    """Keep the LAST run for each (freq_low, freq_high) pair."""
    by_key = {}
    for r in runs:
        key = (_round_key(r['freq_low']), _round_key(r['freq_high']))
        by_key[key] = r    # last wins
    return list(by_key.values())


# ════════════════════════════════════════════════════════════
# Matrix assembly
# ════════════════════════════════════════════════════════════

def build_matrix(runs: list, metric: str):
    """
    Build a square (or rectangular) matrix for `metric`.
    Rows (y) = sorted unique freq_high (ascending = low->high sinusoid, plotted
               bottom-to-top). Cols (x) = sorted unique freq_low.
    Returns (matrix, x_vals, y_vals, coverage_count).
    """
    x_vals = sorted({_round_key(r['freq_low'])  for r in runs})
    y_vals = sorted({_round_key(r['freq_high']) for r in runs})
    xi = {v: i for i, v in enumerate(x_vals)}
    yi = {v: i for i, v in enumerate(y_vals)}

    M = np.full((len(y_vals), len(x_vals)), np.nan)
    filled = 0
    for r in runs:
        if metric not in r:
            continue
        val = r[metric]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        col = xi[_round_key(r['freq_low'])]
        row = yi[_round_key(r['freq_high'])]
        M[row, col] = val
        filled += 1
    return M, x_vals, y_vals, filled


def save_heatmap(M, x_vals, y_vals, metric, dataset, out_dir: Path,
                 higher_better: bool):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.6 * len(x_vals) + 3, 1.4 * len(y_vals) + 3))
    cmap = 'viridis' if higher_better else 'viridis_r'

    # origin='lower' so y ascends upward (low sinusoid at bottom, high at top)
    im = ax.imshow(M, cmap=cmap, aspect='auto', origin='lower')

    ax.set_xticks(range(len(x_vals)))
    ax.set_yticks(range(len(y_vals)))
    ax.set_xticklabels([f'{v:g}' for v in x_vals], rotation=45, ha='right')
    ax.set_yticklabels([f'{v:g}' for v in y_vals])
    ax.set_xlabel('freq_multiplier_low  (LL / convective bulk)   low → high sinusoid →')
    ax.set_ylabel('freq_multiplier_high  (HF / turbulence)   low → high sinusoid →')

    # annotate each cell
    finite = M[np.isfinite(M)]
    if finite.size:
        vmin, vmax = finite.min(), finite.max()
        # mark best cell
        best_idx = (np.nanargmax(M) if higher_better else np.nanargmin(M))
        best_rc = np.unravel_index(best_idx, M.shape)
    else:
        vmin = vmax = 0
        best_rc = (-1, -1)

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if np.isfinite(M[r, c]):
                is_best = (r, c) == best_rc
                txt = f'{M[r, c]:.4f}'
                ax.text(c, r, txt, ha='center', va='center', fontsize=8,
                        color=('white' if is_best else 'black'),
                        fontweight=('bold' if is_best else 'normal'),
                        bbox=(dict(facecolor='red', alpha=0.35, pad=1.5,
                                   edgecolor='none') if is_best else None))
            else:
                ax.text(c, r, '—', ha='center', va='center', fontsize=9,
                        color='gray')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric + ('  (higher better)' if higher_better else '  (lower better)'))

    arrow = '↑' if higher_better else '↓'
    ax.set_title(f'{dataset.upper()} — {metric} {arrow}\n'
                 f'best = {(finite.max() if higher_better else finite.min()):.4f}'
                 if finite.size else f'{dataset.upper()} — {metric} (no data)',
                 fontsize=11)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_metric = metric.replace('/', '_').replace(' ', '_')
    fig_path = out_dir / f'matrix_{dataset}_{safe_metric}.png'
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def write_long_csv(runs: list, metrics: list, dataset: str, out_path: Path):
    import csv
    fields = ['dataset', 'freq_low', 'freq_high'] + metrics + ['source_file']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in sorted(runs, key=lambda d: (d['freq_low'], d['freq_high'])):
            row = {'dataset': dataset, 'freq_low': r['freq_low'],
                   'freq_high': r['freq_high'], 'source_file': r.get('source_file', '')}
            for m in metrics:
                row[m] = r.get(m, '')
            w.writerow(row)
    print(f"  long-format CSV -> {out_path}")


def write_matrix_csv(M, x_vals, y_vals, metric, dataset, out_dir: Path):
    import csv
    safe_metric = metric.replace('/', '_').replace(' ', '_')
    out_path = out_dir / f'matrix_{dataset}_{safe_metric}.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['freq_high\\freq_low'] + [f'{v:g}' for v in x_vals])
        # write top row = highest freq_high, matching origin='lower' visual
        for ri in range(len(y_vals) - 1, -1, -1):
            w.writerow([f'{y_vals[ri]:g}'] +
                       [(f'{M[ri, ci]:.6f}' if np.isfinite(M[ri, ci]) else '')
                        for ci in range(len(x_vals))])


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--logs_glob', nargs='+', required=True,
                   help="One or more glob patterns to log.log files. "
                        "CIKM: 'Exps/multiseed_cikm/*/logs/log.log'. "
                        "Shanghai: a single concatenated log path.")
    p.add_argument('--dataset', required=True)
    p.add_argument('--thresholds', type=int, nargs='+', default=[35, 40],
                   help='Per-threshold CSI values to extract (default: 35 40).')
    p.add_argument('--metrics', nargs='+', default=None,
                   help='Which metrics to plot. Default: CSI-M, per-threshold CSIs, '
                        'HSS, MSE, SSIM, PSNR.')
    p.add_argument('--out_dir', default='gabor_matrices')
    p.add_argument('--no_plots', action='store_true',
                   help='Only write CSVs, skip heatmap PNGs.')
    return p


def main():
    args = build_parser().parse_args()

    # Expand globs
    log_paths = []
    for pattern in args.logs_glob:
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"  [WARN] no files matched: {pattern}")
        log_paths.extend(hits)
    if not log_paths:
        raise SystemExit("No log files found. Check --logs_glob.")

    print(f"Found {len(log_paths)} log file(s).")
    runs = collect_runs(log_paths, args.thresholds)
    print(f"Total raw test runs parsed: {len(runs)}")

    runs = deduplicate(runs)
    print(f"After de-dup (unique freq pairs): {len(runs)}")

    if not runs:
        raise SystemExit("No complete test runs parsed. "
                         "Check that logs contain both 'model parameters :' and "
                         "'Test Results :' lines.")

    # Decide metric list
    threshold_metrics = [f'CSI-{t}' for t in args.thresholds]
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = ['CSI-M'] + threshold_metrics + ['HSS', 'MSE', 'SSIM', 'PSNR']

    # register higher-is-better for per-threshold CSI
    for tm in threshold_metrics:
        HIGHER_IS_BETTER.setdefault(tm, True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Long CSV of everything (all metrics per run)
    all_metric_cols = ['CSI-M'] + threshold_metrics + \
                      ['CSI-pool-4x4', 'CSI-pool-16x16', 'HSS', 'MSE', 'MAE',
                       'RMSE', 'PSNR', 'SSIM', 'CRPS', 'LPIPS']
    write_long_csv(runs, all_metric_cols, args.dataset, out_dir / f'all_runs_{args.dataset}.csv')

    # Per-metric matrices
    print(f"\nBuilding matrices for: {', '.join(metrics)}")
    for metric in metrics:
        M, x_vals, y_vals, filled = build_matrix(runs, metric)
        total = len(x_vals) * len(y_vals)
        hib = HIGHER_IS_BETTER.get(metric, True)
        print(f"  {metric:<14} grid {len(y_vals)}x{len(x_vals)}  "
              f"filled {filled}/{total} cells", end='')
        if filled < total:
            print(f"   [!] {total - filled} empty cell(s)")
        else:
            print()
        write_matrix_csv(M, x_vals, y_vals, metric, args.dataset, out_dir)
        if not args.no_plots and filled > 0:
            fig_path = save_heatmap(M, x_vals, y_vals, metric, args.dataset,
                                    out_dir, hib)
            print(f"                 heatmap -> {fig_path}")

    print(f"\nDone. Outputs in: {out_dir}/")


if __name__ == '__main__':
    main()