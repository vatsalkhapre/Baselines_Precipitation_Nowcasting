"""
W&B logging of the Gabor operator (Experiment 1).

Three distinct quantities are logged and never conflated:

  A. RAW SINUSOID            gabor/<subband>/sinusoid/response
                             a CURVE of sin(z) over the fixed probe range
  B. SINUSOID FREQUENCY      gabor/<subband>/frequency/{mean,std,min,max}
                             gabor/<subband>/effective_frequency/{mean,std,min,max}
                             (scalar parameter summaries; effective = freq_multiplier * freq)
  C. COMPLETE GABOR RESPONSE gabor/<subband>/gabor_response
                             a CURVE of sin(z) * exp(-0.5 * D * gamma)

Plus per-subband scalar summaries (mean/std/min/max) of
freq, effective_frequency, gamma, mu, linear.weight, linear.bias
and periodic histograms of freq and gamma.
"""

import os

import numpy as np
import torch

from .gabor_probe import probe_gabor_layer, select_neurons


def _stats(name, tensor):
    t = tensor.detach().float().reshape(-1)
    return {
        f'{name}/mean': t.mean().item(),
        f'{name}/std': t.std(unbiased=False).item(),
        f'{name}/min': t.min().item(),
        f'{name}/max': t.max().item(),
    }


def gabor_parameter_scalars(model):
    """
    (B) + parameter summaries for every Gabor layer / wavelet subband.
    Large tensors are reduced to mean/std/min/max -- no per-element scalars.
    """
    log = {}
    for sub, layer in model.gabor_layers().items():
        p = f'gabor/{sub}'
        eff = layer.freq_multiplier * layer.freq
        log.update(_stats(f'{p}/freq', layer.freq))
        log.update(_stats(f'{p}/effective_frequency', eff))
        log.update(_stats(f'{p}/gamma', layer.gamma))
        log.update(_stats(f'{p}/mu', layer.mu))
        log.update(_stats(f'{p}/linear_weight', layer.linear.weight))
        log.update(_stats(f'{p}/linear_bias', layer.linear.bias))
        log[f'{p}/freq_multiplier'] = float(layer.freq_multiplier)
    return log


def gabor_histograms(model, wandb):
    """Periodic histograms of freq and gamma (only these two)."""
    log = {}
    for sub, layer in model.gabor_layers().items():
        p = f'gabor/{sub}'
        log[f'{p}/freq/hist'] = wandb.Histogram(
            layer.freq.detach().float().cpu().numpy())
        log[f'{p}/gamma/hist'] = wandb.Histogram(
            layer.gamma.detach().float().cpu().numpy())
    return log


def gabor_probe_curves(model, s, x_probe, num_neurons):
    """
    Evaluate the fixed deterministic probe on every Gabor subband.

    Returns {subband: {'neurons': [...], 's': (P,),
                       'sinusoid'/'envelope'/'gabor': (P, K)}}
    """
    out = {}
    s_np = s.detach().cpu().numpy()
    for sub, layer in model.gabor_layers().items():
        resp = probe_gabor_layer(layer, x_probe)
        neurons = select_neurons(layer.linear.out_features, num_neurons)
        out[sub] = {
            'neurons': neurons,
            's': s_np,
            'sinusoid': resp['sinusoid'][:, neurons],
            'envelope': resp['envelope'][:, neurons],
            'gabor': resp['gabor'][:, neurons],
            'freq': layer.freq.detach().float().cpu().numpy()[neurons],
            'effective_frequency': (layer.freq_multiplier
                                    * layer.freq.detach().float().cpu().numpy()[neurons]),
        }
    return out


def mean_curve_summaries(curves):
    """
    Scalar summaries of the probe curves, one number per subband per quantity.

    These exist because W&B cannot overlay *images* from two runs on one chart,
    but it overlays scalars trivially -- so these are what make RANDOM vs STORM
    directly comparable inside W&B.

    'mean_abs' / 'rms' measure the magnitude of the operator and do not suffer
    sign cancellation across neurons; 'neuron_mean_abs' is the magnitude of the
    neuron-averaged curve, so the ratio neuron_mean_abs / mean_abs says how much
    the neurons of a subband agree in phase (1.0 = fully aligned, 0 = cancel).
    """
    log = {}
    for sub, d in curves.items():
        p = f'gabor/{sub}'
        for name, key in (('gabor_response', 'gabor'),
                          ('sinusoid', 'sinusoid'),
                          ('envelope', 'envelope')):
            a = d[key]                                    # (P, K)
            log[f'{p}/{name}/mean_abs'] = float(np.abs(a).mean())
            log[f'{p}/{name}/rms'] = float(np.sqrt((a ** 2).mean()))
            log[f'{p}/{name}/peak_abs'] = float(np.abs(a).max())
        g = d['gabor']
        log[f'{p}/gabor_response/neuron_mean_abs'] = float(np.abs(g.mean(1)).mean())
        log[f'{p}/gabor_response/neuron_std'] = float(g.std(1).mean())
        denom = float(np.abs(g).mean())
        log[f'{p}/gabor_response/phase_alignment'] = (
            float(np.abs(g.mean(1)).mean() / denom) if denom > 0 else 0.0)
    return log


def save_probe_npz(curves, path, tag):
    """Persist raw probe curves locally for post-hoc analysis."""
    os.makedirs(path, exist_ok=True)
    payload = {}
    for sub, d in curves.items():
        payload[f'{sub}/s'] = d['s']
        payload[f'{sub}/neurons'] = np.asarray(d['neurons'])
        for k in ('sinusoid', 'envelope', 'gabor', 'freq', 'effective_frequency'):
            payload[f'{sub}/{k}'] = d[k]
    fp = os.path.join(path, f'gabor_probe_{tag}.npz')
    np.savez_compressed(fp, **payload)
    return fp


def gabor_state_dict(model):
    """
    All Gabor parameters organised by wavelet subband (saved as gabor_state.pt).
    Kept for future experiments -- nothing is transferred or frozen here.
    """
    state = {}
    for sub, layer in model.gabor_layers().items():
        state[sub] = {
            'freq': layer.freq.detach().cpu().clone(),
            'effective_frequency': (layer.freq_multiplier
                                    * layer.freq.detach().cpu().clone()),
            'gamma': layer.gamma.detach().cpu().clone(),
            'mu': layer.mu.detach().cpu().clone(),
            'linear.weight': layer.linear.weight.detach().cpu().clone(),
            'linear.bias': layer.linear.bias.detach().cpu().clone(),
            'freq_multiplier': float(layer.freq_multiplier),
            'in_features': int(layer.linear.in_features),
            'out_features': int(layer.linear.out_features),
        }
    return state
