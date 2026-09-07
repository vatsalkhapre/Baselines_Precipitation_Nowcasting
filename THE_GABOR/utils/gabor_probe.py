"""
Fixed deterministic Gabor probe (Experiment 1).

The probe is constructed from constants only -- no RNG, no training data, no
device- or run-dependent state -- so it is bit-identical across RANDOM/STORM,
pixel/latent, every epoch and every seed.

Construction
------------
A Gabor layer maps R^{T_in} -> R^{T_out}: its input feature dimension is the
input temporal length.  The probe therefore sweeps a straight line through the
origin of R^{T_in}:

    s      = linspace(-span, +span, num_points)          (the plot x-axis)
    u      = ones(T_in) / sqrt(T_in)                     (fixed unit direction)
    x_probe[p] = s[p] * u                                shape (num_points, T_in)

`||x_probe[p]|| = |s[p]|`, so the x-axis is signed distance from the origin
along a fixed direction.

Three quantities are evaluated on that probe, kept strictly separate:

  A. raw sinusoid            sin(z),         z = freq_multiplier * freq * linear(x)
  B. sinusoid frequency      freq  and  effective_frequency = freq_multiplier * freq
  C. complete Gabor response sin(z) * exp(-0.5 * D(x) * gamma)

with D(x) the squared-distance term of the layer itself.
"""

import math

import numpy as np
import torch

# Probe defaults -- treated as constants of the experiment.
PROBE_NUM_POINTS = 201
PROBE_SPAN = 3.0


def build_probe(t_in, num_points=PROBE_NUM_POINTS, span=PROBE_SPAN,
                dtype=torch.float32):
    """Returns (s, x_probe) with s: (P,) and x_probe: (P, t_in), both on CPU."""
    s = torch.linspace(-span, span, num_points, dtype=dtype)
    u = torch.ones(t_in, dtype=dtype) / math.sqrt(t_in)
    x_probe = s[:, None] * u[None, :]
    return s, x_probe


def select_neurons(out_features, num_neurons):
    """
    Deterministic neuron selection: evenly spaced indices over the output
    dimension.  Depends only on (out_features, num_neurons), so RANDOM and
    STORM always probe exactly the same neurons.
    """
    k = min(num_neurons, out_features)
    idx = np.unique(np.linspace(0, out_features - 1, k).round().astype(int))
    return [int(i) for i in idx]


@torch.no_grad()
def probe_gabor_layer(layer, x_probe):
    """
    Evaluate one GaborLayer on the fixed probe.

    Returns dict of numpy arrays, each (P, out_features):
        'sinusoid' : sin(z)                                    (A)
        'envelope' : exp(-0.5 * D(x) * gamma)
        'gabor'    : sin(z) * envelope                         (C)
    """
    device = layer.linear.weight.device
    dtype = layer.linear.weight.dtype
    x = x_probe.to(device=device, dtype=dtype)

    z = layer.freq_multiplier * layer.freq * layer.linear(x)
    sinusoid = torch.sin(z)

    # D(x) computed inline (not via a helper) so this works with BOTH the
    # original DAWN-Cast GaborLayer and THE_GABOR's copy -- identical formula.
    D = ((x ** 2).sum(-1)[..., None]
         + (layer.mu ** 2).sum(-1)[None, :]
         - 2 * x @ layer.mu.T)
    envelope = torch.exp(-0.5 * D * layer.gamma[None, :])

    return {
        'sinusoid': sinusoid.float().cpu().numpy(),
        'envelope': envelope.float().cpu().numpy(),
        'gabor': (sinusoid * envelope).float().cpu().numpy(),
    }
