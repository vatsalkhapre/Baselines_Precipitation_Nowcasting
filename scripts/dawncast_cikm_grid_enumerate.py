"""
PART A -- DAWN-Cast CIKM capacity grid: enumeration + hard parameter filter.

Standalone. Reads models/DAWNCast/dawncast.py only through get_model(); does not
modify it, and does not touch run_alphapre_convlstm.py.

Grid (CIKM pixel space only):
    level              : 1, 2, 3, 4
    hidden_size_factor : integer expansion factors (see HSF_CANDIDATES)
    num_blocks         : every integer divisor of hidden_size = hidden_dim * T_out
                         (STRModule asserts hidden_size % num_blocks == 0)

Fixed: hidden_dim=64, hf_mode='separate', size_factor=<resolved default>, k_spatial=3.

Filter: total trainable params <= 15,000,000 (counted for real, by instantiating
the model, not by a closed-form estimate).

Per-dataset parameter ceilings:
    cikm                 : 15,000,000   <-- this script
    shanghai, meteonet   : 55,000,000   (pass --budget 55000000, and swap in
                                         that dataset's T_in/T_out)

Usage:
    python3 scripts/dawncast_cikm_grid_enumerate.py
    python3 scripts/dawncast_cikm_grid_enumerate.py --out_dir scripts --budget 15000000
"""

import argparse
import csv
import inspect
import os
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from models.DAWNCast.dawncast import (  # noqa: E402
    DAWNCast, DAWNCastForecaster, FATBlock, WGTMBlock, get_model,
)

# Hard ceiling on total trainable parameters. CIKM = 15M.
# (Shanghai / MeteoNet are 55M -- override with --budget for those.)
PARAM_BUDGET = 15_000_000

# Integer expansion factors. The repo has no hidden_size_factor scaling study to
# reuse -- the only values that appear anywhere are 1 (run_freq_sweep_cikm.sh,
# eval_dawncast.sh, convert_amplinet_to_dawncast.py), 3 and 4
# (Supplementry_codes/compute_analysis.py, run_freq_sweep_meteonet.sh). So this
# is a chosen range, not a reused one: [1, 2, 3, 4, 6, 8].
# Fractional factors (0.25/0.5) are NOT usable -- STRModule builds its weights as
# torch.randn(..., block_size * hidden_size_factor), which requires an int, and
# run_alphapre_convlstm.py declares --spectral_hidden_size_factor as type=int.
# 6 and 8 are kept in the grid so the upper end is enumerated and rejected
# explicitly rather than never being considered.
HSF_CANDIDATES = [1, 2, 3, 4, 6, 8]
LEVEL_CANDIDATES = [1, 2, 3, 4]

# Fixed architecture knobs (from the task spec).
HIDDEN_DIM = 64
HF_MODE = 'separate'
K_SPATIAL = 3

# Placeholders -- Gabor init values and the STR sparsity threshold are supplied
# at training time (Part B). Step 4 verifies they do not move the param count.
PLACEHOLDER_GABOR = dict(
    weight_scale_low=0.1, alpha_low=1.0, beta_low=1.0, freq_multiplier_low=0.5,
    weight_scale_high=0.1, alpha_high=1.0, beta_high=1.0, freq_multiplier_high=2.0,
)
PLACEHOLDER_SPARSITY = 0.01
PLACEHOLDER_WAVE = 'db4'  # matches scripts/scripts_run/run_freq_sweep_cikm.sh


# ------------------------------------------------------------------
# Step 1 -- resolve size_factor's actual default from the signatures
# ------------------------------------------------------------------
def resolve_size_factor_default():
    defaults = {}
    for fn, name in [(get_model, 'get_model'),
                     (DAWNCastForecaster.__init__, 'DAWNCastForecaster.__init__'),
                     (DAWNCast.__init__, 'DAWNCast.__init__'),
                     (WGTMBlock.__init__, 'WGTMBlock.__init__'),
                     (FATBlock.__init__, 'FATBlock.__init__')]:
        p = inspect.signature(fn).parameters.get('size_factor')
        if p is None:
            defaults[name] = '<absent>'
        elif p.default is inspect.Parameter.empty:
            defaults[name] = '<required, no default>'
        else:
            defaults[name] = p.default

    print("STEP 1 -- resolved size_factor defaults (read from signatures):")
    for name, val in defaults.items():
        print(f"    {name:<34} size_factor = {val!r}")

    resolved = defaults['get_model']
    assert isinstance(resolved, float), f"get_model has no usable size_factor default: {resolved!r}"
    print(f"    -> RESOLVED size_factor default (get_model, the entry point) = {resolved}")
    print("    -> DAWNCastForecaster.__init__ takes size_factor as REQUIRED (no default);")
    print("       get_model is what supplies it, so get_model's default is the operative one.\n")
    return resolved


# ------------------------------------------------------------------
# Step 2 -- CIKM T_in / T_out, read from the repo
# ------------------------------------------------------------------
def cikm_temporal_config(repo_root):
    """Pull CIKM's T_in/T_out from run_alphapre_convlstm.py's hardcoded cikm branch."""
    runner = osp.join(repo_root, 'run_alphapre_convlstm.py')
    src = open(runner).read()
    marker = "if self.args.dataset == 'cikm':"
    idx = src.index(marker)
    snippet = src[idx:idx + 200].splitlines()[:3]

    t_in = t_out = None
    for line in snippet:
        if 'frames_in' in line:
            t_in = int(line.split('=')[-1].strip())
        elif 'frames_out' in line:
            t_out = int(line.split('=')[-1].strip())
    assert t_in is not None and t_out is not None, f"could not parse CIKM branch: {snippet}"

    print("STEP 2 -- CIKM temporal config (from run_alphapre_convlstm.py, cikm branch):")
    for line in snippet:
        print(f"        {line.strip()}")
    print(f"    T_in  = {t_in}")
    print(f"    T_out = {t_out}")
    print(f"    hidden_size = hidden_dim * T_out = {HIDDEN_DIM} * {t_out} = {HIDDEN_DIM * t_out}")
    print("    (cross-check: scripts/scripts_run/run_freq_sweep_cikm.sh uses "
          "FRAMES_IN=5, FRAMES_OUT=10, SEQ_LEN=15, IMG_SIZE=128, IMG_CHANNEL=1)\n")
    return t_in, t_out


# ------------------------------------------------------------------
# Step 3 -- divisors of hidden_size
# ------------------------------------------------------------------
def divisors(n):
    ds = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
        i += 1
    return sorted(ds)


# ------------------------------------------------------------------
# Step 4 -- real trainable parameter count
# ------------------------------------------------------------------
def build_and_count(num_blocks, hsf, level, t_in, t_out, size_factor,
                    wave=PLACEHOLDER_WAVE, sparsity=PLACEHOLDER_SPARSITY,
                    gabor=None):
    gabor = PLACEHOLDER_GABOR if gabor is None else gabor
    model = get_model(
        afno_blocks=num_blocks,
        sparsity_threshold=sparsity,
        afno_hidden_size_factor=hsf,
        size_factor=size_factor,
        k_spatial=K_SPATIAL,
        img_channels=1,
        dim=HIDDEN_DIM,
        T_in=t_in, T_out=t_out,
        wave=wave, wavelet_level=level, hf_mode=HF_MODE,
        input_shape=(128, 128),
        **gabor,
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    del model
    return n


def sanity_check_invariance(t_in, t_out, size_factor):
    """Confirm the placeholder values genuinely do not move the param count."""
    print("STEP 4a -- placeholder-invariance check (num_blocks=2, hsf=1, level=2):")
    base = build_and_count(2, 1, 2, t_in, t_out, size_factor)
    print(f"    baseline (wave={PLACEHOLDER_WAVE}, sparsity={PLACEHOLDER_SPARSITY}, "
          f"placeholder Gabor) = {base:,}")

    variants = [
        ("different Gabor init values",
         dict(gabor=dict(weight_scale_low=7.5, alpha_low=3.0, beta_low=0.17,
                         freq_multiplier_low=714.49, weight_scale_high=0.25,
                         alpha_high=2.0, beta_high=4.81, freq_multiplier_high=95.56))),
        ("different sparsity_threshold (0.5)", dict(sparsity=0.5)),
        ("different wave (haar)", dict(wave='haar')),
        ("different wave (db6)", dict(wave='db6')),
    ]
    all_same = True
    for label, kw in variants:
        n = build_and_count(2, 1, 2, t_in, t_out, size_factor, **kw)
        same = (n == base)
        all_same &= same
        print(f"    {label:<38} = {n:,}  {'[same]' if same else '[DIFFERS]'}")
    print(f"    -> placeholders are param-count invariant: {all_same}\n")
    return all_same


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget', type=int, default=PARAM_BUDGET,
                    help='hard ceiling on total trainable parameters')
    ap.add_argument('--out_dir', type=str, default=osp.dirname(osp.abspath(__file__)),
                    help='where to write cikm_valid_combos.csv / cikm_rejected_combos.csv')
    ap.add_argument('--skip_invariance_check', action='store_true')
    args = ap.parse_args()

    repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    torch.manual_seed(0)

    print("=" * 78)
    print("PART A -- DAWN-Cast CIKM capacity grid enumeration")
    print("=" * 78 + "\n")

    size_factor = resolve_size_factor_default()
    t_in, t_out = cikm_temporal_config(repo_root)
    hidden_size = HIDDEN_DIM * t_out

    nb_candidates = divisors(hidden_size)
    print("STEP 3 -- candidate grid (CIKM only):")
    print(f"    level              : {LEVEL_CANDIDATES}")
    print(f"    hidden_size_factor : {HSF_CANDIDATES}   "
          "(chosen integer range; no scaling-study list exists in the repo,")
    print("                          and fractional factors break STRModule's "
          "torch.randn(block_size*hsf))")
    print(f"    num_blocks         : all {len(nb_candidates)} divisors of "
          f"hidden_size={hidden_size}:")
    print(f"                         {nb_candidates}")
    total = len(LEVEL_CANDIDATES) * len(HSF_CANDIDATES) * len(nb_candidates)
    print(f"    total combinations : {total}\n")

    if not args.skip_invariance_check:
        sanity_check_invariance(t_in, t_out, size_factor)

    print(f"STEP 4b/5 -- instantiating all {total} models and counting real trainable params")
    print(f"             (hard ceiling {args.budget:,})\n")

    valid, rejected, errors = [], [], []
    for level in LEVEL_CANDIDATES:
        for hsf in HSF_CANDIDATES:
            for nb in nb_candidates:
                try:
                    n = build_and_count(nb, hsf, level, t_in, t_out, size_factor)
                except Exception as e:  # e.g. GroupNorm/divisibility edge cases
                    errors.append((nb, hsf, level, f"{type(e).__name__}: {e}"))
                    print(f"    [ERROR ] nb={nb:<4} hsf={hsf} lvl={level} :: "
                          f"{type(e).__name__}: {e}")
                    continue

                row = dict(num_blocks=nb, hidden_size_factor=hsf, level=level,
                           total_params=n)
                if n > args.budget:
                    over = n - args.budget
                    row['overage'] = over
                    row['overage_pct_of_budget'] = round(100.0 * over / args.budget, 4)
                    rejected.append(row)
                    print(f"    [REJECT] nb={nb:<4} hsf={hsf} lvl={level} | "
                          f"{n:>12,} params | over by {over:>11,} "
                          f"({row['overage_pct_of_budget']:.2f}% of the "
                          f"{args.budget / 1e6:.0f}M CIKM budget)")
                else:
                    valid.append(row)

    valid.sort(key=lambda r: r['total_params'])
    rejected.sort(key=lambda r: r['total_params'])

    os.makedirs(args.out_dir, exist_ok=True)
    valid_csv = osp.join(args.out_dir, 'cikm_valid_combos.csv')
    rej_csv = osp.join(args.out_dir, 'cikm_rejected_combos.csv')

    with open(valid_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['num_blocks', 'hidden_size_factor',
                                          'level', 'total_params'])
        w.writeheader()
        w.writerows(valid)

    with open(rej_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['num_blocks', 'hidden_size_factor', 'level',
                                          'total_params', 'overage',
                                          'overage_pct_of_budget'])
        w.writeheader()
        w.writerows(rejected)

    print("\n" + "=" * 78)
    print("STEP 6 -- summary")
    print("=" * 78)
    print(f"    enumerated       : {total}")
    print(f"    survived (<= {args.budget:,}) : {len(valid)}")
    print(f"    rejected         : {len(rejected)}")
    if errors:
        print(f"    errored          : {len(errors)}")
    if valid:
        lo, hi = valid[0], valid[-1]
        print(f"    min surviving params : {lo['total_params']:,}")
        print(f"    max surviving params : {hi['total_params']:,}")
        print(f"    MAX CAPACITY UNDER BUDGET : num_blocks={hi['num_blocks']}, "
              f"hidden_size_factor={hi['hidden_size_factor']}, level={hi['level']} "
              f"-> {hi['total_params']:,} params "
              f"({args.budget - hi['total_params']:,} under budget)")
        print(f"    MIN CAPACITY              : num_blocks={lo['num_blocks']}, "
              f"hidden_size_factor={lo['hidden_size_factor']}, level={lo['level']} "
              f"-> {lo['total_params']:,} params")
    print(f"\n    wrote {valid_csv}")
    print(f"    wrote {rej_csv}")


if __name__ == '__main__':
    main()
