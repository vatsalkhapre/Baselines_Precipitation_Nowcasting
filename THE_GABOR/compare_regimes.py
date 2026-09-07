"""
Back-fill neuron-mean plots and build RANDOM vs STORM comparisons
from the probe .npz files every run writes.

Why this exists
---------------
W&B overlays *scalars* from two runs on one chart, but it cannot overlay
*images*.  Comparing RANDOM against STORM by eye across two separate image
panels is exactly the awkward part.  This script reads the raw probe arrays
both runs already saved and draws them on shared axes, so the comparison is
one figure instead of two.

It also back-fills the neuron-mean panels for runs that were trained before
those panels existed -- the .npz files hold everything needed to regenerate
them, so no retraining is required.

    # back-fill mean panels for a finished run
    python -m THE_GABOR.compare_regimes --runs Gabor_pixel_SEVIR_random_seed0 --backfill

    # RANDOM vs STORM overlays + divergence, and push it all to one W&B run
    python -m THE_GABOR.compare_regimes \
        --runs Gabor_pixel_SEVIR_random_seed0 Gabor_pixel_SEVIR_storm_seed0 \
        --labels random storm --backfill \
        --wandb_run Gabor_pixel_SEVIR_compare_seed0

Nothing here trains, and nothing here interprets: it only draws what was
measured.
"""

import argparse
import os
import os.path as osp
import re
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from THE_GABOR.utils.experiment import DEFAULT_LOG_ROOT
from THE_GABOR.utils.gabor_logging import mean_curve_summaries
from THE_GABOR.utils.gabor_visualization import (close_figures, make_gabor_figures,
                                                 make_mean_figures, mean_band_figure)

QUANTITIES = (('gabor', 'gabor_response', 'Gabor(x)', 'COMPLETE GABOR RESPONSE'),
              ('sinusoid', 'sinusoid', 'sin(z)', 'RAW SINUSOID'),
              ('envelope', 'envelope', 'envelope', 'GAUSSIAN ENVELOPE'))


# ------------------------------------------------------------------ loading

def tag_order(tag):
    """init -> -1, epoch_007 -> 7, final -> +inf."""
    if tag == 'init':
        return -1
    if tag == 'final':
        return float('inf')
    m = re.match(r'epoch_(\d+)$', tag)
    return int(m.group(1)) if m else float('inf') - 1


def load_run(log_root, run_name):
    """{tag: {subband: {s, neurons, sinusoid, envelope, gabor, freq, ...}}}"""
    probe_dir = osp.join(log_root, run_name, 'gabor_probe')
    if not osp.isdir(probe_dir):
        raise FileNotFoundError(f'no gabor_probe directory for run {run_name}: {probe_dir}')

    out = {}
    for fn in sorted(os.listdir(probe_dir)):
        m = re.match(r'gabor_probe_(.+)\.npz$', fn)
        if not m:
            continue
        tag = m.group(1)
        d = np.load(osp.join(probe_dir, fn))
        curves = {}
        for key in d.files:
            sub, field = key.rsplit('/', 1)
            curves.setdefault(sub, {})[field] = d[key]
        out[tag] = curves
    if not out:
        raise FileNotFoundError(f'no probe .npz files found in {probe_dir}')
    return dict(sorted(out.items(), key=lambda kv: tag_order(kv[0])))


def subbands_of(run):
    first = next(iter(run.values()))
    return list(first.keys())


# ------------------------------------------------------------------ figures

def overlay_figure(runs, labels, tag, sub, key, ylabel, pretty):
    """One quantity, one subband, every regime on shared axes."""
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=120)
    for run, label in zip(runs, labels):
        if tag not in run or sub not in run[tag]:
            continue
        d = run[tag][sub]
        mean_band_figure(d['s'], d[key], '', ylabel, label=label, ax=ax)
    ax.set_xlabel('probe coordinate  s   (x_probe = s * u)')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{sub} | MEAN {pretty} | {tag}', fontsize=9)
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, ncol=len(runs), frameon=False)
    fig.tight_layout()
    return fig


def evolution_figure(runs, labels, sub, metric):
    """One scalar summary vs epoch, every regime on shared axes."""
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=120)
    for run, label in zip(runs, labels):
        xs, ys = [], []
        for tag, curves in run.items():
            if sub not in curves:
                continue
            v = mean_curve_summaries({sub: curves[sub]}).get(f'gabor/{sub}/{metric}')
            if v is None:
                continue
            o = tag_order(tag)
            xs.append(0 if o < 0 else (max(xs) + 1 if o == float('inf') and xs else o))
            ys.append(v)
        if xs:
            ax.plot(xs, ys, lw=1.5, marker='o', ms=2.5, label=label)
    ax.set_xlabel('epoch  (0 = initialisation)')
    ax.set_ylabel(metric)
    ax.set_title(f'{sub} | {metric} over training', fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def divergence_figure(runs, labels, sub, key, pretty):
    """
    L2 distance between the two regimes' neuron-mean curves, per epoch.

    Both arms start from the same initial checkpoint, so this starts at 0 by
    construction; the shape of the rise is the measurement.
    """
    if len(runs) != 2:
        return None
    a, b = runs
    xs, d_mean, d_rms = [], [], []
    common = [t for t in a if t in b]
    for tag in sorted(common, key=tag_order):
        if sub not in a[tag] or sub not in b[tag]:
            continue
        ca, cb = a[tag][sub][key], b[tag][sub][key]
        if ca.shape != cb.shape:
            continue
        o = tag_order(tag)
        xs.append(0 if o < 0 else (max(xs) + 1 if o == float('inf') and xs else o))
        d_mean.append(float(np.sqrt(((ca.mean(1) - cb.mean(1)) ** 2).mean())))
        ra = np.sqrt((ca ** 2).mean(1))
        rb = np.sqrt((cb ** 2).mean(1))
        d_rms.append(float(np.sqrt(((ra - rb) ** 2).mean())))
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=120)
    ax.plot(xs, d_mean, lw=1.5, marker='o', ms=2.5,
            label=f'||mean_{labels[0]} - mean_{labels[1]}||')
    ax.plot(xs, d_rms, lw=1.5, ls='--', marker='s', ms=2.5,
            label=f'||RMS_{labels[0]} - RMS_{labels[1]}||')
    ax.set_xlabel('epoch  (0 = initialisation)')
    ax.set_ylabel('RMS distance between regimes')
    ax.set_title(f'{sub} | {pretty} divergence: {labels[0]} vs {labels[1]}', fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='run names under --log_root')
    ap.add_argument('--labels', nargs='+', default=None, help='defaults to the run names')
    ap.add_argument('--log_root', type=str, default=DEFAULT_LOG_ROOT)
    ap.add_argument('--out_dir', type=str, default=None,
                    help='where comparison figures go (default: <log_root>/_compare)')
    ap.add_argument('--backfill', action='store_true',
                    help='regenerate the neuron-mean panels for each run from its .npz files')
    ap.add_argument('--tags', nargs='+', default=None,
                    help='only these tags for the overlays (default: init, middle, last)')
    ap.add_argument('--wandb_run', type=str, default=None,
                    help='also log every comparison figure to this new W&B run')
    ap.add_argument('--wandb_project', type=str, default='THE_GABOR')
    args = ap.parse_args()

    labels = args.labels or args.runs
    assert len(labels) == len(args.runs), '--labels must match --runs'
    out_dir = args.out_dir or osp.join(args.log_root, '_compare',
                                       '_vs_'.join(labels))
    os.makedirs(out_dir, exist_ok=True)

    runs = []
    for name in args.runs:
        r = load_run(args.log_root, name)
        runs.append(r)
        print(f'[load] {name}: {len(r)} checkpoints '
              f'({list(r)[0]} .. {list(r)[-1]}), subbands={subbands_of(r)}')

    # ---- back-fill the per-run mean panels ----
    if args.backfill:
        for name, run in zip(args.runs, runs):
            plot_dir = osp.join(args.log_root, name, 'gabor_plots')
            n = 0
            for tag, curves in run.items():
                figs = make_gabor_figures(curves, tag, save_dir=plot_dir)
                figs.update(make_mean_figures(curves, tag, save_dir=plot_dir))
                n += len(figs)
                close_figures(figs)
            print(f'[backfill] {name}: wrote {n} mean panels into {plot_dir}')

    # ---- choose tags for the overlays ----
    common_tags = [t for t in runs[0] if all(t in r for r in runs)]
    common_tags.sort(key=tag_order)
    if args.tags:
        tags = [t for t in args.tags if t in common_tags]
    elif len(common_tags) >= 3:
        tags = [common_tags[0], common_tags[len(common_tags) // 2], common_tags[-1]]
    else:
        tags = common_tags
    print(f'[overlay] tags: {tags}  (of {len(common_tags)} common checkpoints)')

    wb = None
    if args.wandb_run:
        import wandb
        wb = wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run,
                   config={'runs': args.runs, 'labels': labels,
                           'comparison_only': True}, dir=out_dir)

    payload = {}
    subs = subbands_of(runs[0])

    # ---- overlays ----
    for tag in tags:
        for sub in subs:
            for key, name, ylabel, pretty in QUANTITIES:
                f = overlay_figure(runs, labels, tag, sub, key, ylabel, pretty)
                d_out = osp.join(out_dir, tag)
                os.makedirs(d_out, exist_ok=True)
                f.savefig(osp.join(d_out, f'{sub}_{name}_overlay.png'))
                if wb:
                    payload[f'compare/{tag}/{sub}/{name}'] = wb.Image(f)
                plt.close(f)

    # ---- evolution of the scalar summaries ----
    metrics = ('gabor_response/mean_abs', 'gabor_response/rms',
               'gabor_response/phase_alignment', 'sinusoid/mean_abs',
               'envelope/mean_abs')
    for sub in subs:
        for metric in metrics:
            f = evolution_figure(runs, labels, sub, metric)
            d_out = osp.join(out_dir, 'evolution')
            os.makedirs(d_out, exist_ok=True)
            f.savefig(osp.join(d_out, f'{sub}_{metric.replace("/", "_")}.png'))
            if wb:
                payload[f'compare/evolution/{sub}/{metric}'] = wb.Image(f)
            plt.close(f)

    # ---- divergence between the two regimes ----
    if len(runs) == 2:
        for sub in subs:
            for key, name, _, pretty in QUANTITIES:
                f = divergence_figure(runs, labels, sub, key, pretty)
                if f is None:
                    continue
                d_out = osp.join(out_dir, 'divergence')
                os.makedirs(d_out, exist_ok=True)
                f.savefig(osp.join(d_out, f'{sub}_{name}_divergence.png'))
                if wb:
                    payload[f'compare/divergence/{sub}/{name}'] = wb.Image(f)
                plt.close(f)

    if wb:
        wb.log(payload, step=0)
        wb.finish()
        print(f'[wandb] logged {len(payload)} comparison figures to {args.wandb_run}')

    print(f'[done] comparison figures in {out_dir}')


if __name__ == '__main__':
    main()
