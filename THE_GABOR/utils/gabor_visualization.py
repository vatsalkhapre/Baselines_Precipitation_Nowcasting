"""
Deterministic Gabor visualisation (Experiment 1).

For the same selected neurons and the same fixed probe, three separate figures
are produced per wavelet subband so that they can be compared directly:

    1. raw sinusoid            sin(z)
    2. sinusoid frequency      freq / effective_frequency (per selected neuron)
    3. complete Gabor response sin(z) * exp(-0.5 * D * gamma)

The Gabor output dimension is the forecast temporal axis: output neuron n
produces predicted frame n (verified -- nothing after the Gabor mixes time:
the fusion is a 1x1x1 Conv3d, the IDWT is spatial, the projection is
frame-wise).  So the neuron axis of these plots is forecast lead time.

Figures are saved locally and returned so the caller can also push them to W&B.
"""

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def _line_figure(s, curves, neurons, title, ylabel):
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=120)
    for j, n in enumerate(neurons):
        ax.plot(s, curves[:, j], lw=1.2, label=f'n={n} (frame {n + 1})')
    ax.set_xlabel('probe coordinate  s   (x_probe = s * u)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=6, ncol=min(len(neurons), 4), frameon=False)
    fig.tight_layout()
    return fig


def _bar_figure(neurons, freq, eff_freq, title):
    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=120)
    x = np.arange(len(neurons))
    ax.bar(x - 0.2, freq, width=0.4, label='freq')
    ax.bar(x + 0.2, eff_freq, width=0.4, label='effective_frequency')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n(frame {n + 1})' for n in neurons], fontsize=7)
    ax.set_xlabel('Gabor output neuron n  =  predicted frame index')
    ax.set_ylabel('frequency parameter')
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig


def mean_band_figure(s, curves, title, ylabel, label=None, ax=None):
    """
    Neuron-averaged view of one probe quantity.

    Shows three things, because the plain mean alone can be misleading:
      * mean across neurons          (solid)
      * +/- 1 std across neurons     (shaded band)
      * RMS across neurons           (dashed) -- magnitude that cannot cancel

    Neurons of a subband can be in opposite phase, in which case the mean
    collapses toward zero while the RMS does not; keeping both on one axes
    makes that visible instead of hiding it.
    """
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=120)
    else:
        fig = ax.figure

    m = curves.mean(axis=1)
    sd = curves.std(axis=1)
    rms = np.sqrt((curves ** 2).mean(axis=1))
    pre = f'{label} ' if label else ''

    band = ax.fill_between(s, m - sd, m + sd, alpha=0.20, lw=0)
    color = band.get_facecolor()[0][:3]
    ax.plot(s, m, lw=1.6, color=color, label=f'{pre}mean')
    ax.plot(s, rms, lw=1.2, ls='--', color=color, label=f'{pre}RMS')

    if own:
        ax.set_xlabel('probe coordinate  s   (x_probe = s * u)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
    return fig


def make_mean_figures(curves, tag, save_dir=None):
    """
    Neuron-averaged companion to `make_gabor_figures`: one panel per subband
    per quantity instead of K overlapping per-neuron traces.
    """
    figures = {}
    spec = (('gabor', 'gabor_response', 'Gabor(x)', 'COMPLETE GABOR RESPONSE'),
            ('sinusoid', 'sinusoid', 'sin(z)', 'RAW SINUSOID'),
            ('envelope', 'envelope', 'envelope', 'GAUSSIAN ENVELOPE'))
    for sub, d in curves.items():
        s, neurons = d['s'], d['neurons']
        for key, name, ylabel, pretty in spec:
            f = mean_band_figure(
                s, d[key],
                f'{sub} | MEAN {pretty} over {len(neurons)} probe neurons | {tag}',
                ylabel)
            figures[f'gabor/{sub}/{name}/mean'] = f
            if save_dir is not None:
                d_out = os.path.join(save_dir, tag)
                os.makedirs(d_out, exist_ok=True)
                f.savefig(os.path.join(d_out, f'{sub}_{name}_mean.png'))
    return figures


def make_gabor_figures(curves, tag, save_dir=None):
    """
    curves : output of gabor_logging.gabor_probe_curves
    tag    : checkpoint label, e.g. 'init', 'epoch_005', 'final'

    Returns {wandb_key: matplotlib figure}.  Callers must close the figures.
    """
    figures = {}
    for sub, d in curves.items():
        s, neurons = d['s'], d['neurons']

        f1 = _line_figure(s, d['sinusoid'], neurons,
                          f'{sub} | RAW SINUSOID  sin(z) | {tag}', 'sin(z)')
        f2 = _bar_figure(neurons, d['freq'], d['effective_frequency'],
                         f'{sub} | SINUSOID FREQUENCY | {tag}')
        f3 = _line_figure(s, d['gabor'], neurons,
                          f'{sub} | COMPLETE GABOR RESPONSE  sin(z)*exp(-0.5*D*gamma) | {tag}',
                          'Gabor(x)')
        f4 = _line_figure(s, d['envelope'], neurons,
                          f'{sub} | GAUSSIAN ENVELOPE  exp(-0.5*D*gamma) | {tag}',
                          'envelope')

        figures[f'gabor/{sub}/sinusoid/response'] = f1
        figures[f'gabor/{sub}/frequency/selected_neurons'] = f2
        figures[f'gabor/{sub}/gabor_response'] = f3
        figures[f'gabor/{sub}/envelope_response'] = f4

        if save_dir is not None:
            d_out = os.path.join(save_dir, tag)
            os.makedirs(d_out, exist_ok=True)
            f1.savefig(os.path.join(d_out, f'{sub}_sinusoid.png'))
            f2.savefig(os.path.join(d_out, f'{sub}_frequency.png'))
            f3.savefig(os.path.join(d_out, f'{sub}_gabor_response.png'))
            f4.savefig(os.path.join(d_out, f'{sub}_envelope.png'))

    return figures


def close_figures(figures):
    for f in figures.values():
        plt.close(f)
