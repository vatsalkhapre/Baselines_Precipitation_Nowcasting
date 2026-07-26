#!/usr/bin/env python3
"""
gabor_theta_convergence.py
────────────────────────────
Question this answers: after training, do the LEARNED sine arguments (theta)
in the LL and HF FAT blocks converge toward similar values regardless of
which (freq_low, freq_high) cell they started from — or does each cell stay
distinguishable, i.e. does the initial freq_multiplier leave a lasting mark?

Two cases, both loaded and compared:
    Case 1: original GaborLayer  (gamma trainable, nn.Parameter)
    Case 2: frozen GaborLayer    (gamma non-trainable, register_buffer)

IMPORTANT CONFOUND (from diffing your two model files):
    Case 1's GaborLayer still uses the OLD bias init (Uniform(-pi, pi),
    unscaled by gamma) — it was never updated with the gamma-matched bias
    fix. Case 2 has BOTH the frozen-gamma change AND the bias fix. So any
    difference you see between Case 1 and Case 2 is a mix of
    "gamma trainable vs frozen" AND "buggy bias vs fixed bias" — not a
    clean single-variable ablation. Keep this in mind when interpreting
    results; this script reports what happened, it can't separate the two
    causes for you.

Why gamma itself is irrelevant to theta:
    z = freq_multiplier * freq * (W @ x + b)
    gamma appears ONLY in the Gaussian envelope exp(-0.5*gamma*D(x)), never
    in the sine's argument. So "does theta converge" is a question about the
    LEARNED freq, W, and bias only. Whether gamma was trainable affects this
    only indirectly (through what gradients W/freq/bias received during
    training), never directly.

freq_multiplier is NOT saved in the checkpoint:
    In GaborLayer.__init__, `self.freq_multiplier = freq_multiplier` is a
    plain Python float, not an nn.Parameter or buffer — it will not appear
    in state_dict(). Each checkpoint's (freq_low, freq_high) must come from
    an external source: this script tries, in order,
      1. a sibling `logs/log.log` next to the checkpoint (same convention
         gabor_sweep_matrix.py uses), reading the "model parameters :" line
      2. a regex applied to the checkpoint's own path, trying both known
         naming conventions (CIKM: freq_X_Y_cikm_betas_...,
         Shanghai: Shanghai_flowX_fhighY), or a user-supplied --path_regex
         override if the checkpoint location changed and neither matches.

Usage
─────
    python gabor_theta_convergence.py \
        --run dataset=cikm     case=1 ckpt_glob='Exps/multiseed_cikm_case1/*/checkpoints/ckpt-best.pt' \
        --run dataset=cikm     case=2 ckpt_glob='Exps/multiseed_cikm_case2/*/checkpoints/ckpt-best.pt' \
        --run dataset=shanghai case=1 ckpt_glob='Exps/gabor_exp_shanghai_case1/*/checkpoints/ckpt-best.pt' \
        --run dataset=shanghai case=2 ckpt_glob='Exps/gabor_exp_shanghai_case2/*/checkpoints/ckpt-best.pt' \
        --out_dir theta_convergence_matrices

    # if checkpoints moved and neither default naming pattern matches:
    --run dataset=cikm case=1 ckpt_glob='...' path_regex='mytag_(?P<freq_low>[\\d.]+)_(?P<freq_high>[\\d.]+)'

Requirements: torch, numpy, matplotlib
"""

import argparse
import ast
import glob
import re
from pathlib import Path

import numpy as np
import torch


# ════════════════════════════════════════════════════════════
# freq_low / freq_high extraction
# ════════════════════════════════════════════════════════════

RE_MODEL_PARAMS = re.compile(r"model parameters\s*:?\s*(\{.*\})\s*$")

# Known directory-naming conventions from your .sh scripts, tried in order.
DEFAULT_PATH_REGEXES = [
    r"freq_(?P<freq_low>[\d.]+)_(?P<freq_high>[\d.]+)_cikm_betas",          # CIKM
    r"Shanghai_flow(?P<freq_low>[\d.]+)_fhigh(?P<freq_high>[\d.]+)",        # Shanghai
    r"Meteonet_flow(?P<freq_low>[\d.]+)_fhigh(?P<freq_high>[\d.]+)",        # Meteonet
    r"flow(?P<freq_low>[\d.]+)_fhigh(?P<freq_high>[\d.]+)",                 # Generic flow/fhigh
    r"freq_(?P<freq_low>[\d.]+)_(?P<freq_high>[\d.]+)",                     # Generic freq_low/freq_high
]


def extract_freqs_from_log(log_path: Path):
    """Read the LAST 'model parameters :' dict in the log and pull the freqs."""
    if not log_path.exists():
        return None
    with open(log_path, 'r', errors='replace') as f:
        text = f.read()
    matches = RE_MODEL_PARAMS.findall(text)
    if not matches:
        return None
    try:
        d = ast.literal_eval(matches[-1])
        return float(d['freq_multiplier_low']), float(d['freq_multiplier_high'])
    except (ValueError, SyntaxError, KeyError):
        return None


def extract_freqs_from_path(ckpt_path: Path, extra_regex: str = None):
    regexes = ([extra_regex] if extra_regex else []) + DEFAULT_PATH_REGEXES
    s = str(ckpt_path)
    for pattern in regexes:
        m = re.search(pattern, s)
        if m:
            return float(m.group('freq_low')), float(m.group('freq_high'))
    return None


def extract_freqs(ckpt_path: Path, path_regex: str = None):
    """Try sibling log.log first, then path-based regex. Raises if both fail."""
    # Sibling log convention: .../<exp_name>/checkpoints/ckpt-best.pt
    #                         .../<exp_name>/logs/log.log
    log_path = ckpt_path.parent.parent / 'logs' / 'log.log'
    freqs = extract_freqs_from_log(log_path)
    if freqs is not None:
        return freqs, f'log:{log_path}'

    freqs = extract_freqs_from_path(ckpt_path, path_regex)
    if freqs is not None:
        return freqs, 'path_regex'

    raise ValueError(
        f"Could not determine freq_low/freq_high for {ckpt_path}. "
        f"Tried sibling log at {log_path} (not found or unparseable), and "
        f"default path regexes. Pass path_regex=... in --run to override."
    )


# ════════════════════════════════════════════════════════════
# State-dict loading and Gabor-band discovery
# ════════════════════════════════════════════════════════════

def load_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ('model', 'model_state_dict', 'state_dict', 'net'):
            if key in ckpt:
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        raise ValueError(f"Cannot locate state dict in {ckpt_path}. "
                         f"Top-level keys: {list(ckpt.keys())}")
    raise TypeError(f"Unexpected checkpoint type in {ckpt_path}: {type(ckpt)}")


def discover_bands(sd: dict) -> dict:
    """
    Find every GaborLayer's state-dict prefix in this checkpoint.
    Returns {'LL': prefix, 'HF0': prefix, 'HF1': prefix, ...} — HF count is
    auto-detected (handles hf_mode='shared' -> single 'HF' entry, or
    hf_mode='separate' -> HF0..HF{level-1}).

    Matches the module hierarchy confirmed from your model files:
        lastocast.operator.stream_ll.gabor
        lastocast.operator.stream_hf.gabor              (hf_mode='shared')
        lastocast.operator.hf_streams.{i}.gabor          (hf_mode='separate')
    """
    roots = ['lastocast.operator', 'operator']
    bands = {}

    for root in roots:
        ll_prefix = f'{root}.stream_ll.gabor'
        if f'{ll_prefix}.freq' in sd:
            bands['LL'] = ll_prefix
            break

    for root in roots:
        shared_prefix = f'{root}.stream_hf.gabor'
        if f'{shared_prefix}.freq' in sd:
            bands['HF'] = shared_prefix
            break
    else:
        # try separate hf_streams, auto-detect count by probing indices
        for root in roots:
            i = 0
            found_any = False
            while True:
                prefix = f'{root}.hf_streams.{i}.gabor'
                if f'{prefix}.freq' in sd:
                    bands[f'HF{i}'] = prefix
                    found_any = True
                    i += 1
                else:
                    break
            if found_any:
                break

    if 'LL' not in bands:
        gabor_keys = sorted({k.rsplit('.gabor.', 1)[0] + '.gabor'
                             for k in sd if '.gabor.' in k})
        raise ValueError(f"Could not find LL band. Gabor-like prefixes found: {gabor_keys}")

    return bands


# ════════════════════════════════════════════════════════════
# Theta reconstruction from LEARNED weights
# ════════════════════════════════════════════════════════════

def learned_theta_stats(sd: dict, prefix: str, freq_multiplier: float,
                        n_trials: int = 4000, x_std: float = 1.0,
                        real_latents: torch.Tensor = None,
                        seed: int = 0) -> dict:
    """
    Reconstruct the ACTUAL sine argument z = freq_multiplier * freq * (W@x + b)
    using this checkpoint's LEARNED freq/W/bias (not re-initialized values),
    and report both the full (Wx+b) statistics and a signal-only (Wx) /
    bias-only breakdown, consistent with the variance decomposition
    established earlier in this analysis (bias contributes non-trivially,
    not just Wx).
    """
    def get(key):
        full = f'{prefix}.{key}'
        if full not in sd:
            raise KeyError(f"Key not found: {full}")
        return sd[full].cpu().float()

    freq = get('freq')                  # (out_features,)
    W    = get('linear.weight')         # (out_features, in_features)
    b    = get('linear.bias')           # (out_features,)
    in_features = W.shape[1]

    torch.manual_seed(seed)
    if real_latents is not None:
        idx = torch.randint(0, real_latents.shape[0], (n_trials,))
        x = real_latents[idx]
        if x.shape[1] != in_features:
            raise ValueError(f"real_latents has T_in={x.shape[1]}, "
                             f"but this GaborLayer expects in_features={in_features}")
    else:
        x = torch.randn(n_trials, in_features) * x_std

    lin_signal_only = x @ W.T                      # (n_trials, out_features), no bias
    lin_full        = lin_signal_only + b[None, :]  # actual forward() computation

    z_signal = freq_multiplier * freq[None, :] * lin_signal_only
    z_full   = freq_multiplier * freq[None, :] * lin_full
    sin_full = torch.sin(z_full)

    # bias-only "operating point" scale: freq_multiplier * freq * b (no x)
    z_bias_only = freq_multiplier * freq * b

    return {
        'z_std_full':    float(z_full.std()),
        'z_mean_full':   float(z_full.mean()),
        'z_std_signal_only': float(z_signal.std()),
        'z_bias_scale':  float(z_bias_only.abs().mean()),
        'sin_mean':      float(sin_full.mean()),
        'sin_std':       float(sin_full.std()),
        'sin_min':       float(sin_full.min()),
        'sin_max':       float(sin_full.max()),
        'freq_learned_mean': float(freq.mean()),
        'W_norm_mean':   float(W.norm(dim=1).mean()),
        'bias_abs_mean': float(b.abs().mean()),
    }


# ════════════════════════════════════════════════════════════
# Run collection
# ════════════════════════════════════════════════════════════

def collect_run(dataset: str, case: str, ckpt_glob: str, path_regex: str,
                n_trials: int, x_std: float, seed: int) -> list:
    ckpt_paths = sorted(glob.glob(ckpt_glob))
    if not ckpt_paths:
        print(f"  [WARN] no checkpoints matched: {ckpt_glob}")
        return []

    records = []
    for ckpt_str in ckpt_paths:
        ckpt_path = Path(ckpt_str)
        try:
            (freq_low, freq_high), src = extract_freqs(ckpt_path, path_regex)
        except ValueError as e:
            print(f"  [SKIP] {ckpt_path}: {e}")
            continue

        try:
            sd = load_state_dict(ckpt_path)
            bands = discover_bands(sd)
        except (ValueError, TypeError) as e:
            print(f"  [SKIP] {ckpt_path}: {e}")
            continue

        for band_name, prefix in bands.items():
            fm = freq_low if band_name == 'LL' else freq_high
            try:
                stats = learned_theta_stats(sd, prefix, fm, n_trials=n_trials,
                                            x_std=x_std, seed=seed)
            except KeyError as e:
                print(f"  [SKIP] {ckpt_path} band={band_name}: {e}")
                continue
            rec = {'dataset': dataset, 'case': case, 'band': band_name,
                   'freq_low': freq_low, 'freq_high': freq_high,
                   'freq_source': src, 'ckpt_path': str(ckpt_path)}
            rec.update(stats)
            records.append(rec)

        print(f"  parsed {ckpt_path.parent.parent.name}  "
              f"(freq_low={freq_low:g}, freq_high={freq_high:g})  "
              f"bands={list(bands.keys())}  [src={src}]")

    return records


# ════════════════════════════════════════════════════════════
# Matrix assembly (reused pattern, no "best cell" — purely descriptive)
# ════════════════════════════════════════════════════════════

def _round_key(x, ndigits=2):
    return round(float(x), ndigits)


def build_matrix(records: list, metric: str):
    x_vals = sorted({_round_key(r['freq_low'])  for r in records})
    y_vals = sorted({_round_key(r['freq_high']) for r in records})
    xi = {v: i for i, v in enumerate(x_vals)}
    yi = {v: i for i, v in enumerate(y_vals)}
    M = np.full((len(y_vals), len(x_vals)), np.nan)
    for r in records:
        if metric not in r:
            continue
        col, row = xi[_round_key(r['freq_low'])], yi[_round_key(r['freq_high'])]
        M[row, col] = r[metric]
    return M, x_vals, y_vals


def coefficient_of_variation(M: np.ndarray) -> float:
    finite = M[np.isfinite(M)]
    if finite.size < 2 or finite.mean() == 0:
        return float('nan')
    return float(finite.std() / abs(finite.mean()))


def save_heatmap(M, x_vals, y_vals, metric, dataset, case, band, out_dir: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.6*len(x_vals)+3, 1.4*len(y_vals)+3))
    im = ax.imshow(M, cmap='magma', aspect='auto', origin='lower')

    ax.set_xticks(range(len(x_vals))); ax.set_yticks(range(len(y_vals)))
    ax.set_xticklabels([f'{v:g}' for v in x_vals], rotation=45, ha='right')
    ax.set_yticklabels([f'{v:g}' for v in y_vals])
    ax.set_xlabel('freq_multiplier_low  (LL init)   low → high sinusoid →')
    ax.set_ylabel('freq_multiplier_high  (HF init)   low → high sinusoid →')

    for r in range(M.shape[0]):
        for c in range(M.shape[1]):
            if np.isfinite(M[r, c]):
                ax.text(c, r, f'{M[r,c]:.3f}', ha='center', va='center',
                        fontsize=8, color='white')
            else:
                ax.text(c, r, '—', ha='center', va='center', fontsize=9, color='gray')

    cv = coefficient_of_variation(M)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(metric)
    ax.set_title(f'{dataset.upper()} — Case {case} — {band} — LEARNED {metric}\n'
                 f'coefficient of variation across grid = {cv:.3f}  '
                 f'(low = converged, high = still path-dependent)', fontsize=10)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f'theta_{dataset}_case{case}_{band}_{metric}.png'
    fig.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def write_matrix_csv(M, x_vals, y_vals, metric, dataset, case, band, out_dir: Path):
    import csv
    out_path = out_dir / f'theta_{dataset}_case{case}_{band}_{metric}.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['freq_high\\freq_low'] + [f'{v:g}' for v in x_vals])
        for ri in range(len(y_vals)-1, -1, -1):
            w.writerow([f'{y_vals[ri]:g}'] +
                      [(f'{M[ri,ci]:.6f}' if np.isfinite(M[ri,ci]) else '')
                       for ci in range(len(x_vals))])


def write_long_csv(records: list, out_path: Path):
    import csv
    fields = ['dataset', 'case', 'band', 'freq_low', 'freq_high', 'freq_source',
              'z_std_full', 'z_mean_full', 'z_std_signal_only', 'z_bias_scale',
              'sin_mean', 'sin_std', 'sin_min', 'sin_max',
              'freq_learned_mean', 'W_norm_mean', 'bias_abs_mean', 'ckpt_path']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in sorted(records, key=lambda d: (d['dataset'], d['case'], d['band'],
                                                d['freq_low'], d['freq_high'])):
            w.writerow(r)


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def parse_run_arg(items: list) -> dict:
    """Parse ['dataset=cikm', 'case=1', "ckpt_glob=..."] into a dict."""
    d = {}
    for item in items:
        if '=' not in item:
            raise argparse.ArgumentTypeError(f"--run item must be key=value, got: {item}")
        k, v = item.split('=', 1)
        d[k] = v
    for req in ('dataset', 'case', 'ckpt_glob'):
        if req not in d:
            raise argparse.ArgumentTypeError(f"--run missing required key '{req}': {items}")
    return d


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', nargs='+', action='append', required=True,
                   help="One run group: dataset=<name> case=<1|2> ckpt_glob=<pattern> "
                        "[path_regex=<pattern>]. Repeat --run for each dataset x case "
                        "combination.")
    p.add_argument('--metrics', nargs='+',
                   default=['z_std_full', 'z_mean_full', 'sin_std'],
                   help='Which reconstructed-theta metrics to build matrices for.')
    p.add_argument('--n_mc_trials', type=int, default=4000)
    p.add_argument('--x_std', type=float, default=1.0)
    p.add_argument('--real_latents_path', default=None,
                   help='Optional .pt/.npy of shape (N, T_in) with real AE latents, '
                        'used for ALL runs instead of synthetic Gaussian noise.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_dir', default='theta_convergence_matrices')
    p.add_argument('--no_plots', action='store_true')
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    real_latents = None
    if args.real_latents_path:
        path = Path(args.real_latents_path)
        real_latents = (torch.from_numpy(np.load(path)).float() if path.suffix == '.npy'
                        else torch.load(path, map_location='cpu').float())
        print(f"Using real latents: {tuple(real_latents.shape)}")

    all_records = []
    for run_items in args.run:
        run = parse_run_arg(run_items)
        print(f"\n--- dataset={run['dataset']}  case={run['case']} ---")
        recs = collect_run(run['dataset'], run['case'], run['ckpt_glob'],
                           run.get('path_regex'), args.n_mc_trials, args.x_std, args.seed)
        all_records.extend(recs)

    if not all_records:
        raise SystemExit("No checkpoints parsed across any --run group.")

    write_long_csv(all_records, out_dir / 'all_theta_records.csv')
    print(f"\nLong CSV -> {out_dir / 'all_theta_records.csv'}")

    datasets = sorted({r['dataset'] for r in all_records})
    cases    = sorted({r['case']    for r in all_records})
    bands    = sorted({r['band']    for r in all_records})

    print(f"\nBuilding matrices for datasets={datasets} cases={cases} bands={bands} "
         f"metrics={args.metrics}")
    print(f"\n{'dataset':<10}{'case':<6}{'band':<6}{'metric':<16}{'CV':>8}   (low CV = converged)")
    for dataset in datasets:
        for case in cases:
            for band in bands:
                subset = [r for r in all_records
                         if r['dataset']==dataset and r['case']==case and r['band']==band]
                if not subset:
                    continue
                for metric in args.metrics:
                    M, x_vals, y_vals = build_matrix(subset, metric)
                    cv = coefficient_of_variation(M)
                    print(f"{dataset:<10}{case:<6}{band:<6}{metric:<16}{cv:>8.3f}")
                    write_matrix_csv(M, x_vals, y_vals, metric, dataset, case, band, out_dir)
                    if not args.no_plots:
                        save_heatmap(M, x_vals, y_vals, metric, dataset, case, band, out_dir)

    print(f"\nDone. Outputs in: {out_dir}/")


if __name__ == '__main__':
    main()
