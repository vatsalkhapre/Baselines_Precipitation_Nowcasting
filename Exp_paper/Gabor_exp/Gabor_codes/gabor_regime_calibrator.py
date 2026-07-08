#!/usr/bin/env python3
"""
gabor_regime_calibrator.py
────────────────────────────
Given a dataset's (weight_scale, learned_gamma) per band, this script:

  1. Derives `beta` so that Gamma(alpha, beta) is centered at the learned
     gamma value extracted from your checkpoint (via gabor_checkpoint_inspector.py).
  2. Calibrates 5 freq_multiplier (lambda) values per band, spanning a
     physically-grounded ladder from near-linear to a half-cycle oscillation.
  3. Empirically cross-checks every level by instantiating the ACTUAL
     GaborLayer (with frozen gamma) and measuring realized sin(theta)
     statistics from a real forward pass — not just the analytical estimate.

Why an analytical formula alone is not enough
──────────────────────────────────────────────
The mean-field target  z ≈ λ · f_mean · E[‖W‖]  only uses the MEAN of gamma
and freq. But with alpha=1, Gamma(1, beta) is an EXPONENTIAL distribution
(coefficient of variation = 1 — very heavy-tailed), and freq ~ Uniform(0,1)
has its own spread. Both contribute extra variance beyond their means, so
the empirically realized z_std is systematically LARGER than the naive
target (verified ~1.5-2x higher in practice). The empirical cross-check
table below is what actually tells you what regime you'll get — treat the
analytical z_target as a design anchor, and the empirical columns as ground
truth.

Includes the frozen-gamma GaborLayer
─────────────────────────────────────
  - self.gamma is now a registered buffer (not nn.Parameter) -> no gradient,
    never touched by the optimizer, still moves with .to(device) and
    appears in state_dict.
  - bias uses the gamma-matched scaling you already changed to
    (weight_scale * sqrt(gamma)), fixing the earlier bias-dominance bug.

Usage
─────
    python gabor_regime_calibrator.py \
        --dataset cikm \
        --weight_scale_low 0.1  --gamma_learned_low 0.0232 \
        --weight_scale_high 0.25 --gamma_learned_high 0.2075 \
        --t_in 5 --t_out 10 \
        --save_csv

Requirements: torch, numpy
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn


# ════════════════════════════════════════════════════════════
# 1. Modified GaborLayer — gamma frozen, gamma-matched bias
# ════════════════════════════════════════════════════════════

class GaborLayer(nn.Module):
    """
    Gabor activation with FROZEN gamma (bandwidth fixed at init, not
    learned) and gamma-matched bias scaling.

    Changes from the original:
      - self.gamma is a registered buffer, not an nn.Parameter.
        -> requires_grad is never True; the optimizer never sees it;
           it still saves/loads correctly via state_dict and .to(device).
      - bias = Uniform(-1,1) * weight_scale * sqrt(gamma)  (already the
        version you're using) instead of the old fixed Uniform(-pi, pi),
        which kept bias's std IDENTICAL in form to E[||W||], so bias no
        longer dominates the sine argument regardless of gamma/weight_scale.
    """
    def __init__(self, in_features, out_features, weight_scale,
                 alpha=1.0, beta=1.0, freq_multiplier=1.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)

        gamma = torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        self.register_buffer('gamma', gamma)      # <-- frozen: buffer, not Parameter

        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data = (2 * torch.rand(out_features) - 1) * weight_scale * torch.sqrt(self.gamma)

        self.freq = nn.Parameter(torch.rand(out_features))
        self.freq_multiplier = freq_multiplier

    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.freq_multiplier * self.freq * self.linear(x)) * \
               torch.exp(-0.5 * D * self.gamma[None, :])


# ════════════════════════════════════════════════════════════
# 2. Beta derivation and freq_multiplier calibration
# ════════════════════════════════════════════════════════════

def derive_beta(alpha: float, learned_gamma_mean: float) -> float:
    """
    Gamma(alpha, beta) [PyTorch rate parameterisation] has E[gamma] = alpha/beta.
    Solve beta so a freshly-sampled distribution is centered at the value
    your checkpoint's Gabor layer actually converged to.
    """
    return alpha / learned_gamma_mean


# z targets: standard deviation of the sine argument, chosen at physically
# meaningful points on sin(z) itself — not picked arbitrarily.
#   0.10  -> sin(z) deviates 0.2%  from linear    (deep Regime A)
#   0.30  -> sin(z) deviates 1.5%                 (still near-linear)
#   0.80  -> sin(z) deviates 10.3%                (visibly curved / transitional)
#   1.60  -> sits at sin's peak (~pi/2)            (max single-lobe nonlinearity)
#   pi    -> completes a half-cycle, sin crosses 0 (genuinely oscillatory)
Z_TARGETS = [0.10, 0.30, 0.80, 1.60, math.pi]
Z_LABELS  = ['near-linear', 'mild', 'transitional',
             'peak-curvature (~pi/2)', 'half-cycle (~pi)']
F_INIT_MEAN = 0.5   # E[freq], freq ~ Uniform(0,1)


def expected_W_norm(weight_scale: float, gamma_mean: float) -> float:
    """
    E[||W_j||] for a GaborLayer neuron after weight_scale*sqrt(gamma) scaling.

    Derivation: nn.Linear default (Kaiming-uniform) init gives
    W_ij ~ Uniform(-1/sqrt(in), 1/sqrt(in)), so Var(W_ij) = 1/(3*in).
    Summing over `in` components: E[||W_row||^2] = in * 1/(3*in) = 1/3,
    independent of in_features. So E[||W_row||] ~= sqrt(1/3).
    After scaling by weight_scale*sqrt(gamma):
        E[||W_row||] ~= weight_scale * sqrt(gamma) * sqrt(1/3)
                      = weight_scale * sqrt(gamma / 3)
    """
    return weight_scale * math.sqrt(gamma_mean / 3.0)


def calibrate_freq_multiplier(weight_scale: float, gamma_mean: float,
                               z_targets=Z_TARGETS, f_mean=F_INIT_MEAN) -> list:
    """
    freq_multiplier values so the mean-field std of the sine argument
    (driven by input variation; x assumed ~unit-variance post-AE latent)
    hits each z_target:
        z = freq_multiplier * f_mean * E[||W||]
        => freq_multiplier = z / (f_mean * E[||W||])

    NOTE: this is a design anchor, not a precise prediction — see the
    empirical cross-check below for the actually realized values.
    """
    EW = expected_W_norm(weight_scale, gamma_mean)
    denom = f_mean * EW
    return [z / denom for z in z_targets]


# ════════════════════════════════════════════════════════════
# 3. Empirical cross-check — actual forward pass, actual layer
# ════════════════════════════════════════════════════════════

def empirical_sin_stats(t_in: int, t_out: int, weight_scale: float,
                        alpha: float, beta: float, freq_multiplier: float,
                        n_trials: int = 4000, x_std: float = 1.0,
                        real_latents: torch.Tensor = None,
                        seed: int = None) -> dict:
    """
    Instantiate the ACTUAL frozen-gamma GaborLayer with the given
    hyperparameters, run it on input, and measure the realised sine-argument
    and sin(theta) statistics.

    Input source (in priority order):
      1. real_latents, if provided: a tensor of shape (N, t_in) sampled from
         your ACTUAL encoded AE latents. This is the only way to capture the
         true shape (skew, tails, bounds) of your data — reflectivity is
         non-negative and typically right-skewed, so a synthetic Gaussian
         may match the variance but not the tail behavior, which is exactly
         what drives the 'full range reached' flag in the report.
      2. synthetic torch.randn(n_trials, t_in) * x_std, if real_latents is
         None. x_std defaults to 1.0 (standard normal): ~68% of values in
         [-1,1], ~99.7% in [-3,3]. VERIFY this against your real latent std
         before trusting the cross-check numbers — this is an assumption,
         not a measurement, unless you supply real_latents.
    """
    if seed is not None:
        torch.manual_seed(seed)

    layer = GaborLayer(t_in, t_out, weight_scale, alpha=alpha, beta=beta,
                       freq_multiplier=freq_multiplier)
    layer.eval()

    with torch.no_grad():
        if real_latents is not None:
            idx = torch.randint(0, real_latents.shape[0], (n_trials,))
            x = real_latents[idx]
        else:
            x = torch.randn(n_trials, t_in) * x_std
        z = layer.freq_multiplier * layer.freq * layer.linear(x)   # (n_trials, t_out)
        sin_theta = torch.sin(z)

    return {
        'z_std':    float(z.std()),
        'z_min':    float(z.min()),
        'z_max':    float(z.max()),
        'sin_mean': float(sin_theta.mean()),
        'sin_std':  float(sin_theta.std()),
        'sin_min':  float(sin_theta.min()),
        'sin_max':  float(sin_theta.max()),
        'gamma_realized_mean': float(layer.gamma.mean()),
        'input_source': 'real_latents' if real_latents is not None else f'synthetic N(0,{x_std}^2)',
        'input_x_std_used': float(x.std()),   # actual realized std, whichever source
    }


# ════════════════════════════════════════════════════════════
# 4. Full report builder
# ════════════════════════════════════════════════════════════

def build_calibration_report(dataset_name: str,
                              weight_scale_low: float, gamma_learned_low: float,
                              weight_scale_high: float, gamma_learned_high: float,
                              alpha: float = 1.0,
                              t_in: int = 5, t_out: int = 20,
                              n_mc_trials: int = 4000,
                              x_std: float = 1.0,
                              real_latents: torch.Tensor = None,
                              seed: int = 0) -> dict:
    """
    Full calibration + empirical cross-check for one dataset's LL and HF bands.

    x_std / real_latents: see empirical_sin_stats docstring. Passing
    real_latents (shape (N, t_in), sampled from your actual encoded AE
    outputs) is strongly preferred over the synthetic Gaussian default —
    reflectivity-derived latents are typically right-skewed, not symmetric.
    """
    beta_low  = derive_beta(alpha, gamma_learned_low)
    beta_high = derive_beta(alpha, gamma_learned_high)

    lam_low  = calibrate_freq_multiplier(weight_scale_low,  gamma_learned_low)
    lam_high = calibrate_freq_multiplier(weight_scale_high, gamma_learned_high)

    rows = []
    for i, (z_t, label, ll, lh) in enumerate(zip(Z_TARGETS, Z_LABELS, lam_low, lam_high)):
        ll_stats = empirical_sin_stats(t_in, t_out, weight_scale_low,  alpha, beta_low,  ll,
                                       n_trials=n_mc_trials, x_std=x_std,
                                       real_latents=real_latents, seed=seed + i)
        hf_stats = empirical_sin_stats(t_in, t_out, weight_scale_high, alpha, beta_high, lh,
                                       n_trials=n_mc_trials, x_std=x_std,
                                       real_latents=real_latents, seed=seed + 100 + i)
        rows.append({
            'level': f'L{i}', 'z_target': z_t, 'regime': label,
            'freq_multiplier_low': ll, 'freq_multiplier_high': lh,
            'LL': ll_stats, 'HF': hf_stats,
        })

    return {
        'dataset': dataset_name, 'alpha': alpha,
        't_in': t_in, 't_out': t_out,
        'beta_low': beta_low, 'beta_high': beta_high,
        'gamma_learned_low': gamma_learned_low, 'gamma_learned_high': gamma_learned_high,
        'weight_scale_low': weight_scale_low, 'weight_scale_high': weight_scale_high,
        'freq_multiplier_low': lam_low, 'freq_multiplier_high': lam_high,
        'input_source': rows[0]['LL']['input_source'] if rows else 'n/a',
        'rows': rows,
    }


# ════════════════════════════════════════════════════════════
# 5. Printing
# ════════════════════════════════════════════════════════════

def print_report(report: dict):
    W = 100
    print(f"\n{'='*W}")
    print(f"  GABOR REGIME CALIBRATION — {report['dataset'].upper()}")
    print(f"{'='*W}")
    print(f"  alpha = {report['alpha']}   T_in = {report['t_in']}   T_out = {report['t_out']}")
    print(f"  weight_scale_low  = {report['weight_scale_low']}   "
          f"learned gamma_LL = {report['gamma_learned_low']}")
    print(f"  weight_scale_high = {report['weight_scale_high']}   "
          f"learned gamma_HF = {report['gamma_learned_high']}")
    print(f"\n  Derived beta_low  = {report['beta_low']:.4f}   "
          f"(Gamma(1,beta) mean = {1.0/report['beta_low']:.4f})")
    print(f"  Derived beta_high = {report['beta_high']:.4f}   "
          f"(Gamma(1,beta) mean = {1.0/report['beta_high']:.4f})")

    print(f"\n  {'-'*W}")
    print(f"  DESIGN TARGET (analytical, mean-field — use as a starting anchor)")
    print(f"  {'-'*W}")
    print(f"  {'Level':<5} {'z target':>9}  {'regime':<24} {'freq_mult_low':>14} {'freq_mult_high':>15}")
    for r in report['rows']:
        print(f"  {r['level']:<5} {r['z_target']:>9.3f}  {r['regime']:<24} "
              f"{r['freq_multiplier_low']:>14.2f} {r['freq_multiplier_high']:>15.2f}")

    print(f"\n  {'-'*W}")
    print(f"  EMPIRICAL CROSS-CHECK  (actual GaborLayer forward pass)")
    print(f"  Input source: {report['input_source']}")
    print(f"  {'-'*W}")
    print(f"  {'Level':<5} {'band':<4} {'z target':>8} {'realized':>9} {'sin mean':>9} "
          f"{'sin std':>8} {'sin min':>8} {'sin max':>8}  {'full range?':<12}")
    for r in report['rows']:
        for band in ('LL', 'HF'):
            s = r[band]
            full_range = 'YES' if (s['sin_min'] < -0.95 and s['sin_max'] > 0.95) else 'no'
            print(f"  {r['level']:<5} {band:<4} {r['z_target']:>8.3f} {s['z_std']:>9.3f} "
                  f"{s['sin_mean']:>9.3f} {s['sin_std']:>8.3f} {s['sin_min']:>8.3f} "
                  f"{s['sin_max']:>8.3f}  {full_range:<12}")

    print(f"\n  Reading guide:")
    print(f"    'realized' (empirical z_std) is typically LARGER than 'z target' — this is")
    print(f"    expected: gamma (Exponential when alpha=1, CV=1) and freq (Uniform(0,1)) both")
    print(f"    contribute extra per-neuron variance beyond their means, which the analytical")
    print(f"    formula does not capture. Trust this empirical table over the design target.")
    print(f"    'full range?' = YES means at least one neuron/sample reached sin ≈ ±1 — the")
    print(f"    tails of the heavy-tailed gamma distribution can saturate individual neurons")
    print(f"    even while the population's TYPICAL (sin std) response is still modest.")
    print(f"{'='*W}\n")


def write_csv(report: dict, out_path: Path):
    import csv
    fields = ['dataset', 'level', 'z_target', 'regime',
              'freq_multiplier_low', 'freq_multiplier_high',
              'LL_z_std', 'LL_sin_mean', 'LL_sin_std', 'LL_sin_min', 'LL_sin_max',
              'HF_z_std', 'HF_sin_mean', 'HF_sin_std', 'HF_sin_min', 'HF_sin_max']
    write_header = not out_path.exists()
    with open(out_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if write_header:
            w.writeheader()
        for r in report['rows']:
            row = {
                'dataset': report['dataset'], 'level': r['level'],
                'z_target': r['z_target'], 'regime': r['regime'],
                'freq_multiplier_low': r['freq_multiplier_low'],
                'freq_multiplier_high': r['freq_multiplier_high'],
            }
            for band in ('LL', 'HF'):
                for k, v in r[band].items():
                    if k in ('z_std', 'sin_mean', 'sin_std', 'sin_min', 'sin_max'):
                        row[f'{band}_{k}'] = v
            w.writerow(row)
    print(f"  CSV appended -> {out_path}")


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True)
    p.add_argument('--weight_scale_low',   type=float, required=True)
    p.add_argument('--gamma_learned_low',  type=float, required=True)
    p.add_argument('--weight_scale_high',  type=float, required=True)
    p.add_argument('--gamma_learned_high', type=float, required=True)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--t_in',  type=int, default=5)
    p.add_argument('--t_out', type=int, default=20)
    p.add_argument('--n_mc_trials', type=int, default=4000)
    p.add_argument('--x_std', type=float, default=1.0,
                   help='Std of synthetic input if --real_latents_path not given. '
                        'DEFAULT 1.0 IS AN ASSUMPTION (matches N(0,1): ~68%% in '
                        '[-1,1], ~99.7%% in [-3,3]) — verify against your real '
                        'AE latents rather than trusting this default.')
    p.add_argument('--real_latents_path', default=None,
                   help='Path to a .pt or .npy file of shape (N, t_in) containing '
                        'REAL encoded AE latent values (e.g. sampled from your '
                        'training set). Strongly preferred over synthetic Gaussian '
                        'noise, since reflectivity-derived latents are typically '
                        'right-skewed, not symmetric — the tail shape directly '
                        'affects the "full range reached" flag in the report.')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save_csv', action='store_true')
    p.add_argument('--out_dir', default='.')
    return p


def main():
    args = build_parser().parse_args()

    real_latents = None
    if args.real_latents_path is not None:
        path = Path(args.real_latents_path)
        if path.suffix == '.npy':
            real_latents = torch.from_numpy(np.load(path)).float()
        else:
            real_latents = torch.load(path, map_location='cpu').float()
        assert real_latents.ndim == 2 and real_latents.shape[1] == args.t_in, (
            f"--real_latents_path must be shape (N, t_in={args.t_in}), "
            f"got {tuple(real_latents.shape)}"
        )
        print(f"  Loaded real latents: {tuple(real_latents.shape)}  "
              f"mean={real_latents.mean():.4f}  std={real_latents.std():.4f}  "
              f"min={real_latents.min():.4f}  max={real_latents.max():.4f}")

    report = build_calibration_report(
        dataset_name=args.dataset,
        weight_scale_low=args.weight_scale_low, gamma_learned_low=args.gamma_learned_low,
        weight_scale_high=args.weight_scale_high, gamma_learned_high=args.gamma_learned_high,
        alpha=args.alpha, t_in=args.t_in, t_out=args.t_out,
        n_mc_trials=args.n_mc_trials, x_std=args.x_std,
        real_latents=real_latents, seed=args.seed,
    )
    print_report(report)
    if args.save_csv:
        write_csv(report, Path(args.out_dir) / 'gabor_regime_calibration.csv')


if __name__ == '__main__':
    main()