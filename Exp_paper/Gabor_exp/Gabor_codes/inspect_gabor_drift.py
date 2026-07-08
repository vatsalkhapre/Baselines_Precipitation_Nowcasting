#!/usr/bin/env python3
"""
gabor_checkpoint_inspector.py
──────────────────────────────
Diagnose learned Gabor parameters (gamma, freq) from a WaveletLASTOCast
or DAWNCast checkpoint. Compares final values to the initialization
distribution to answer: does init dominate, or have gradients moved the
parameters into a different regime?

Decision rule
─────────────
  |drift| < 1σ      → INIT_DOMINATES   → sweep (beta, freq_multiplier) at init
  1σ ≤ |drift| < 3σ → MIXED           → sweep init AND log realized values each run
  |drift| ≥ 3σ      → LEARNED_DOMINATES → freeze gamma/freq; use learned means as grid

Usage examples (matching your screenshot configs)
─────────────────────────────────────────────────
# Shanghai  (beta=0.17, f_low=4.0, f_high=4.0)
python gabor_checkpoint_inspector.py \
    --ckpt /path/to/shanghai_best.pt --dataset shanghai \
    --beta_low 0.17  --freq_multiplier_low  4.0 --weight_scale_low  1.0 \
    --beta_high 0.17 --freq_multiplier_high 4.0 --weight_scale_high 1.0

# CIKM  (beta=100, f_low=0.1, f_high=0.1)
python gabor_checkpoint_inspector.py \
    --ckpt /path/to/cikm_best.pt --dataset cikm \
    --beta_low 100  --freq_multiplier_low  0.1 --weight_scale_low  1.0 \
    --beta_high 100 --freq_multiplier_high 0.1 --weight_scale_high 1.0

# SEVIR (beta=0.17, f_low=0.1, f_high=4.0)
python gabor_checkpoint_inspector.py \
    --ckpt /path/to/sevir_best.pt --dataset sevir \
    --beta_low 0.17 --freq_multiplier_low 0.1 --weight_scale_low 1.0 \
    --beta_high 0.17 --freq_multiplier_high 4.0 --weight_scale_high 1.0 \
    --level 2

Requirements: torch, numpy, matplotlib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────
DRIFT_INIT_THRESHOLD    = 1.0
DRIFT_LEARNED_THRESHOLD = 3.0


# ─────────────────────────────────────────────
# State-dict loading
# ─────────────────────────────────────────────

def load_state_dict(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict):
        for key in ('model', 'model_state_dict', 'state_dict', 'net'):
            if key in ckpt:
                print(f"  [ckpt] state_dict found under key '{key}'")
                return ckpt[key]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        print(f"  [ckpt] available top-level keys: {list(ckpt.keys())}")
        raise ValueError("Cannot locate state dict in checkpoint.")
    raise TypeError(f"Unexpected checkpoint type: {type(ckpt)}")


# ─────────────────────────────────────────────
# Key resolution
# ─────────────────────────────────────────────

def find_gabor_prefix(sd, model_type, band, level_idx, hf_mode):
    if model_type == 'lastocast':
        roots    = ['lastocast.operator', 'operator']
        band_map = {'ll': 'stream_ll.gabor', 'hf_shared': 'stream_hf.gabor'}
        if band.startswith('hf_sep'):
            band_map[band] = f'hf_streams.{level_idx}.gabor'
    else:
        roots    = ['dawncast.wgtm', 'wgtm']
        band_map = {'ll': 'fat_ll.gabor', 'hf_shared': 'fat_hf.gabor'}
        if band.startswith('hf_sep'):
            band_map[band] = f'fat_hf_streams.{level_idx}.gabor'

    suffix = band_map[band]
    for root in roots:
        prefix = f'{root}.{suffix}'
        if f'{prefix}.gamma' in sd:
            return prefix

    gamma_keys = [k for k in sd if k.endswith('.gamma')]
    print(f"  [WARN] prefix not found for band='{band}'. gamma keys: {gamma_keys}")
    return None


# ─────────────────────────────────────────────
# Core statistics
# ─────────────────────────────────────────────

def gabor_stats(sd, prefix, freq_multiplier, alpha, beta, weight_scale, label):
    """
    Extract and analyse a GaborLayer from its state-dict prefix.

    eff_freq (effective oscillation density) = λ · f · ‖W‖  per neuron.

    Init approximation
    ──────────────────
    nn.Linear Kaiming-uniform: W_ij ~ Uniform(-1/√in, 1/√in)
    Scaled by weight_scale · √γ_j  →  E[‖W_j‖²] = weight_scale² · γ_j / 3
    So  E[‖W_j‖] ≈ weight_scale · √(γ_init_mean / 3)
    →   eff_freq_init ≈ λ · 0.5 · weight_scale · √(γ_init_mean / 3)
    """
    def get(key):
        full = f'{prefix}.{key}'
        if full not in sd:
            raise KeyError(f"Key not found: {full}")
        return sd[full].cpu().float()

    gamma = get('gamma')
    freq  = get('freq')
    mu    = get('mu')
    W     = get('linear.weight')
    out_features, in_features = W.shape

    # ── Init stats ──────────────────────────────────────────────────────
    gamma_init_mean = alpha / beta
    gamma_init_std  = np.sqrt(alpha) / beta
    freq_init_mean  = 0.5
    freq_init_std   = 1.0 / np.sqrt(12.0)

    # eff_freq_init: E[λ·f·‖W_j‖] at initialisation
    # E[‖W_j‖] ≈ weight_scale · √(gamma_init_mean / 3)  (see docstring)
    eff_freq_init = freq_multiplier * freq_init_mean * weight_scale * np.sqrt(gamma_init_mean / 3.0)

    # ── Learned gamma ────────────────────────────────────────────────────
    g_np = gamma.numpy()
    gm, gs       = float(gamma.mean()), float(gamma.std())
    gmed         = float(gamma.median())
    gmin, gmax   = float(gamma.min()), float(gamma.max())
    g_drift      = (gm - gamma_init_mean) / (gamma_init_std + 1e-12)

    # ── Learned freq ─────────────────────────────────────────────────────
    f_np   = freq.numpy()
    fm, fs = float(freq.mean()), float(freq.std())
    f_drift = (fm - freq_init_mean) / (freq_init_std + 1e-12)

    # ── Effective oscillation (learned) ──────────────────────────────────
    W_norms  = W.norm(dim=1)
    eff_freq = (freq_multiplier * freq * W_norms).numpy()

    # ── Envelope width ───────────────────────────────────────────────────
    env_width_learned = float((1.0 / gamma.sqrt()).mean())
    env_width_init    = 1.0 / np.sqrt(gamma_init_mean)
    env_width_drift   = (env_width_learned - env_width_init) / (env_width_init + 1e-12)

    # ── Verdict ──────────────────────────────────────────────────────────
    max_drift = max(abs(g_drift), abs(f_drift))
    if max_drift < DRIFT_INIT_THRESHOLD:
        verdict = 'INIT_DOMINATES'
    elif max_drift >= DRIFT_LEARNED_THRESHOLD:
        verdict = 'LEARNED_DOMINATES'
    else:
        verdict = 'MIXED'

    return {
        'label':               label,
        'freq_multiplier':     freq_multiplier,
        'weight_scale':        weight_scale,
        'alpha':               alpha,
        'beta':                beta,
        'out_features':        out_features,
        'in_features':         in_features,
        # gamma
        'gamma_init_mean':     gamma_init_mean,
        'gamma_init_std':      gamma_init_std,
        'gamma_learned_mean':  gm,
        'gamma_learned_std':   gs,
        'gamma_learned_med':   gmed,
        'gamma_learned_min':   gmin,
        'gamma_learned_max':   gmax,
        'gamma_drift_sigma':   g_drift,
        'gamma_raw':           g_np,
        # freq
        'freq_init_mean':      freq_init_mean,
        'freq_init_std':       freq_init_std,
        'freq_learned_mean':   fm,
        'freq_learned_std':    fs,
        'freq_drift_sigma':    f_drift,
        'freq_raw':            f_np,
        # effective oscillation
        'eff_freq_init':       eff_freq_init,   # ← analytical init expected value
        'eff_freq_mean':       float(eff_freq.mean()),
        'eff_freq_std':        float(eff_freq.std()),
        'eff_freq_p10':        float(np.percentile(eff_freq, 10)),
        'eff_freq_p50':        float(np.percentile(eff_freq, 50)),
        'eff_freq_p90':        float(np.percentile(eff_freq, 90)),
        'eff_freq_all':        eff_freq,
        # envelope
        'env_width_init':      env_width_init,
        'env_width_learned':   env_width_learned,
        'env_width_drift_pct': env_width_drift * 100.0,
        # mu
        'mu_abs_mean':         float(mu.abs().mean()),
        # verdict
        'verdict':             verdict,
        'max_drift':           max_drift,
    }


# ─────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────

VERDICT_COLOR = {
    'INIT_DOMINATES':    '✅',
    'MIXED':             '⚠️ ',
    'LEARNED_DOMINATES': '🔴',
}
VERDICT_ADVICE = {
    'INIT_DOMINATES':
        'Parameters barely moved (<1σ). The prior (beta, freq_multiplier) is the\n'
        '    operative control. → Sweep (beta, freq_multiplier) at init time.',
    'MIXED':
        'Partial drift (1–3σ). Init matters but gradient adjusts it.\n'
        '    → Sweep init AND log final gamma/freq per run to verify realized regimes.',
    'LEARNED_DOMINATES':
        'Strong drift (>3σ). Gradient discarded the prior.\n'
        '    → Freeze gamma and freq for the matrix experiment.\n'
        '    → Anchor grid at learned mean(gamma) and mean(freq) values.',
}

def _flag(drift):
    a = abs(drift)
    if a >= DRIFT_LEARNED_THRESHOLD: return '  ← LARGE'
    if a >= DRIFT_INIT_THRESHOLD:    return '  ← moderate'
    return ''


def print_report(stats_list, dataset):
    W = 80
    print('\n' + '═' * W)
    print(f"  GABOR PARAMETER INSPECTION  —  dataset: {dataset.upper()}")
    print('═' * W)
    for s in stats_list:
        print(f"\n{'─' * W}")
        print(f"  Layer : {s['label']}")
        print(f"  Config: λ={s['freq_multiplier']}  W_scale={s['weight_scale']}  "
              f"α={s['alpha']}  β={s['beta']}  "
              f"(neurons={s['out_features']}, in={s['in_features']})")
        print(f"{'─' * W}")

        print(f"\n  γ (bandwidth — controls envelope AND init weight scale)")
        print(f"    Init:    E[γ]={s['gamma_init_mean']:.4f}   Std={s['gamma_init_std']:.4f}")
        print(f"    Learned: mean={s['gamma_learned_mean']:.4f}  std={s['gamma_learned_std']:.4f}  "
              f"median={s['gamma_learned_med']:.4f}  "
              f"[{s['gamma_learned_min']:.3f}, {s['gamma_learned_max']:.3f}]")
        print(f"    Drift:   {s['gamma_drift_sigma']:+.2f}σ{_flag(s['gamma_drift_sigma'])}")
        print(f"    Envelope 1/√γ:  init≈{s['env_width_init']:.4f}  "
              f"learned={s['env_width_learned']:.4f}  Δ={s['env_width_drift_pct']:+.1f}%")

        print(f"\n  f (base frequency, Uniform(0,1) at init)")
        print(f"    Init:    E[f]=0.500   Std=0.289")
        print(f"    Learned: mean={s['freq_learned_mean']:.4f}  std={s['freq_learned_std']:.4f}")
        print(f"    Drift:   {s['freq_drift_sigma']:+.2f}σ{_flag(s['freq_drift_sigma'])}")

        print(f"\n  Effective oscillation  λ·f·‖W‖  (sine-arg scale per unit input)")
        print(f"    Init expected ≈ {s['eff_freq_init']:.4f}  "
              f"(λ·0.5·W_scale·√(γ_init/3))")
        print(f"    Learned:  mean={s['eff_freq_mean']:.4f}  std={s['eff_freq_std']:.4f}")
        print(f"    Percentiles: p10={s['eff_freq_p10']:.4f}  "
              f"p50={s['eff_freq_p50']:.4f}  p90={s['eff_freq_p90']:.4f}")

        icon = VERDICT_COLOR[s['verdict']]
        print(f"\n  {icon}  VERDICT: {s['verdict']}")
        print(f"    {VERDICT_ADVICE[s['verdict']]}")

    # Summary table
    print(f"\n{'═' * W}")
    print(f"  {'Layer':<35} {'γ drift':>9} {'f drift':>9} {'eff_freq':>9}  Verdict")
    print(f"  {'─'*35} {'─'*9} {'─'*9} {'─'*9}  {'─'*16}")
    for s in stats_list:
        print(f"  {s['label']:<35} "
              f"{s['gamma_drift_sigma']:>+9.2f}σ "
              f"{s['freq_drift_sigma']:>+9.2f}σ "
              f"{s['eff_freq_mean']:>9.4f}  "
              f"{VERDICT_COLOR[s['verdict']]} {s['verdict']}")
    print(f"{'═' * W}\n")


# ─────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────

def save_figure(stats_list, dataset, ckpt_name, out_dir):
    """
    Per-neuron dot plot.

    gamma, freq, eff_freq are all 1-D tensors of length T_out (e.g. 20 values
    for most datasets, 10 for CIKM).  Each index j is one output neuron of the
    GaborLayer's linear projection (T_in → T_out).  There are only T_out values,
    so a histogram is meaningless — we plot each value as an individual dot
    on the x-axis = neuron index j.

    NOTE: whether j maps strictly to forecast lead time T+j+1 depends on
    downstream IDWT/SRST shuffling, so the x-axis is labelled "output neuron j"
    not "lead time."
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  [WARN] matplotlib not available — skipping figure.")
        return

    C_DOT    = '#4C8BE8'   # blue  — one dot per learned neuron value
    C_INIT   = '#E05252'   # red   — horizontal init reference line
    C_MEAN   = '#F5A623'   # orange — horizontal learned-mean line

    n   = len(stats_list)
    fig = plt.figure(figsize=(17, 4.5 * n))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.70, wspace=0.42)

    for row, s in enumerate(stats_list):
        n_neurons = s['out_features']          # = T_out  (e.g. 20 or 10)
        js        = np.arange(n_neurons)       # x-axis: neuron indices 0 … T_out-1

        # ── shared helper ────────────────────────────────────────────────
        def _dot_plot(ax, values, init_ref, learned_mean,
                      ylabel, title, init_label, mean_label):
            """
            Scatter one dot per neuron.
            Red dashed  = init expected value (analytical).
            Orange solid = learned mean across all neurons.
            """
            ax.scatter(js, values, color=C_DOT, s=55, zorder=3,
                       label='Learned value  (one dot = one output neuron j)')
            ax.axhline(init_ref, color=C_INIT, lw=1.8, ls='--',
                       label=init_label)
            ax.axhline(learned_mean, color=C_MEAN, lw=1.8, ls='-',
                       label=mean_label)
            ax.set_xlabel('Output neuron index j  (0 … T_out−1)', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=8.5, loc='left')
            ax.set_xticks(js)
            ax.grid(axis='y', ls=':', alpha=0.4)
            ax.legend(fontsize=7.5, framealpha=0.88)

        # ── Column 0: γ ─────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 0])
        _dot_plot(
            ax,
            values        = s['gamma_raw'],
            init_ref      = s['gamma_init_mean'],
            learned_mean  = s['gamma_learned_mean'],
            ylabel        = 'γ  (bandwidth)',
            title         = (f'{s["label"]}\n'
                             f'γ  |  init E[γ] = α/β = {s["gamma_init_mean"]:.4f}  '
                             f'|  drift = {s["gamma_drift_sigma"]:+.2f}σ'),
            init_label    = f'Init expected  α/β = {s["gamma_init_mean"]:.4f}',
            mean_label    = f'Learned mean = {s["gamma_learned_mean"]:.4f}',
        )

        # ── Column 1: f ──────────────────────────────────────────────────
        ax = fig.add_subplot(gs[row, 1])
        _dot_plot(
            ax,
            values        = s['freq_raw'],
            init_ref      = s['freq_init_mean'],
            learned_mean  = s['freq_learned_mean'],
            ylabel        = 'f  (base frequency)',
            title         = (f'Base frequency f\n'
                             f'Init ~ Uniform(0,1)  E[f]=0.5  '
                             f'|  drift = {s["freq_drift_sigma"]:+.2f}σ'),
            init_label    = 'Init expected  E[f] = 0.500',
            mean_label    = f'Learned mean = {s["freq_learned_mean"]:.4f}',
        )

        # ── Column 2: λ·f·‖W‖ ───────────────────────────────────────────
        # eff_freq[j] = freq_multiplier * freq[j] * ||W[j,:]||
        # This is the actual scale of the sine argument for neuron j.
        # init_ref = freq_multiplier * 0.5 * weight_scale * sqrt(gamma_init_mean / 3)
        ax = fig.add_subplot(gs[row, 2])
        _dot_plot(
            ax,
            values        = s['eff_freq_all'],
            init_ref      = s['eff_freq_init'],
            learned_mean  = s['eff_freq_mean'],
            ylabel        = 'λ · f · ‖W‖',
            title         = (f'Effective oscillation density  λ·f·‖W‖\n'
                             f'How fast the sine cycles per unit input norm\n'
                             f'Init ≈ {s["eff_freq_init"]:.4f}  '
                             f'→  learned mean = {s["eff_freq_mean"]:.4f}  '
                             f'(Δ = {s["eff_freq_mean"]-s["eff_freq_init"]:+.4f})'),
            init_label    = (f'Init expected ≈ {s["eff_freq_init"]:.4f}\n'
                             f'  λ · 0.5 · W_scale · √(γ₀/3)'),
            mean_label    = f'Learned mean = {s["eff_freq_mean"]:.4f}',
        )

    fig.suptitle(
        f'Gabor parameter analysis — {dataset.upper()} — {ckpt_name}\n'
        f'Each dot = one output neuron j  |  '
        f'Red dashed = init expected value  |  Orange solid = learned mean',
        fontsize=10, y=1.01)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f'gabor_inspect_{dataset}_{ckpt_name}_test1.png'
    plt.savefig(fig_path, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  Figure saved → {fig_path}")


# ─────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────

def write_csv(stats_list, dataset, out_path):
    import csv
    fields = [
        'dataset', 'label', 'freq_multiplier', 'weight_scale', 'alpha', 'beta',
        'gamma_init_mean', 'gamma_init_std',
        'gamma_learned_mean', 'gamma_learned_std',
        'gamma_drift_sigma',
        'freq_init_mean', 'freq_learned_mean', 'freq_drift_sigma',
        'eff_freq_init', 'eff_freq_mean',
        'eff_freq_p10', 'eff_freq_p50', 'eff_freq_p90',
        'env_width_init', 'env_width_learned', 'env_width_drift_pct',
        'mu_abs_mean', 'verdict', 'max_drift',
    ]
    write_header = not out_path.exists()
    with open(out_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if write_header:
            w.writeheader()
        for s in stats_list:
            row = {'dataset': dataset}
            row.update(s)
            w.writerow(row)
    print(f"  CSV appended → {out_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--ckpt',                  required=True)
    p.add_argument('--dataset',               default='unknown')
    p.add_argument('--model_type',            default='lastocast',
                   choices=['lastocast', 'dawncast'])
    p.add_argument('--hf_mode',               default='shared',
                   choices=['shared', 'separate'])
    p.add_argument('--level',                 type=int,   default=1)
    # LL Gabor init config (must match training)
    p.add_argument('--alpha_low',             type=float, default=1.0)
    p.add_argument('--beta_low',              type=float, default=1.0)
    p.add_argument('--freq_multiplier_low',   type=float, default=1.0)
    p.add_argument('--weight_scale_low',      type=float, default=1.5)
    # HF Gabor init config
    p.add_argument('--alpha_high',            type=float, default=1.0)
    p.add_argument('--beta_high',             type=float, default=1.0)
    p.add_argument('--freq_multiplier_high',  type=float, default=1.0)
    p.add_argument('--weight_scale_high',     type=float, default=1.5)
    # Output
    p.add_argument('--save_fig',              action='store_true')
    p.add_argument('--save_csv',              action='store_true')
    p.add_argument('--out_dir',               default='.')
    p.add_argument('--verbose_keys',          action='store_true')
    return p


def main():
    args   = build_parser().parse_args()
    out_dir = Path(args.out_dir)

    print(f"\n{'─'*60}")
    print(f"  Checkpoint : {args.ckpt}")
    print(f"  Dataset    : {args.dataset}   Model: {args.model_type}")
    print(f"  HF mode    : {args.hf_mode}   Level: {args.level}")
    print(f"{'─'*60}\n")

    sd = load_state_dict(args.ckpt)

    if args.verbose_keys:
        gkeys = [k for k in sd if 'gamma' in k or 'freq' in k]
        print("  Gabor-related keys:")
        for k in gkeys:
            print(f"    {k}  {tuple(sd[k].shape)}")
        print()

    stats_list = []

    def collect(band, level_idx, freq_mult, alpha, beta, weight_scale, label):
        prefix = find_gabor_prefix(sd, args.model_type, band, level_idx, args.hf_mode)
        if prefix is None:
            print(f"  [SKIP] prefix not found for '{label}'. Use --verbose_keys.")
            return
        try:
            s = gabor_stats(sd, prefix, freq_mult, alpha, beta, weight_scale, label)
            stats_list.append(s)
            print(f"  [OK] {label}  (prefix={prefix})")
        except KeyError as e:
            print(f"  [SKIP] {label}: {e}")

    # LL
    collect('ll', -1,
            args.freq_multiplier_low, args.alpha_low, args.beta_low,
            args.weight_scale_low,
            'LL  (low-freq / large-scale convection)')

    # HF
    if args.hf_mode == 'shared':
        collect('hf_shared', -1,
                args.freq_multiplier_high, args.alpha_high, args.beta_high,
                args.weight_scale_high,
                'HF shared  (high-freq / turbulence)')
    else:
        tags = ['coarsest'] + ['mid'] * max(args.level - 2, 0) + (['finest'] if args.level > 1 else [])
        for i in range(args.level):
            collect(f'hf_sep_{i}', i,
                    args.freq_multiplier_high, args.alpha_high, args.beta_high,
                    args.weight_scale_high,
                    f'HF level {i}  ({tags[i]})')

    if not stats_list:
        print("\n[ERROR] No layers extracted. Run with --verbose_keys.")
        sys.exit(1)

    print_report(stats_list, args.dataset)

    ckpt_name = Path(args.ckpt).stem

    if args.save_fig:
        save_figure(stats_list, args.dataset, ckpt_name, out_dir)

    if args.save_csv:
        write_csv(stats_list, args.dataset, out_dir / 'gabor_results.csv')

    print("Done.\n")


if __name__ == '__main__':
    main()