# DAWN-Cast Gabor Layer: Changes, Bugs, and Reasoning

A complete record of every Gabor-related decision, code change, and bug fix made in this
thread, with the reasoning behind each one. Organized so it can be read top to bottom or
used as a reference for specific decisions later.

---

## 1. Conceptual foundations: what the two Gabor regimes mean

### 1.1 The climatic-adaptivity hypothesis

The paper frames the Gabor activation as adaptive between two regimes:

- **Regime A (near-linear):** small `γ` (broad envelope) and small `λ·f` (low effective
  frequency) → `sin(z) ≈ z`, so the neuron reduces to a scaled linear projection.
- **Regime B (oscillatory):** large `γ` and large `λ·f` → genuinely non-monotonic,
  periodic response.

The claim is that large-scale convective bulk (LL wavelet band) should prefer Regime A,
and fine-scale turbulence (HF bands) should prefer Regime B.

### 1.2 Why the diagonal (frequency ↔ localization) was assumed — and why it isn't forced

`γ` controls **two things at once**: the Gaussian envelope width (`exp(-0.5·γ·D(x))`) and,
via the `weight_scale·√γ` init coupling, the effective oscillation frequency. The paper's
Regime A/B walks only the **diagonal** of a 2×2 space (low-freq+broad, high-freq+narrow),
not the off-diagonal corners (low-freq+narrow, high-freq+broad).

**Why the diagonal is a defensible default:** it's the **constant-Q locus** — the only
place where dilating a Gabor unit keeps the number of oscillations under its envelope
fixed, which is what makes it behave like a genuine wavelet atom rather than an arbitrary
windowed sinusoid. This is why the model is themed as a "wavelet network."

**Why it isn't a mathematical necessity:** nothing in the forward pass prevents the
off-diagonal corners. The diagonal is what `weight_scale·√γ` produces **at initialization**
— it is not preserved during training (see §1.4). Testing off-diagonal regimes is a valid,
open experiment, not a violation of the architecture.

### 1.3 Correcting a real discrepancy: "spatial" localization is actually temporal

The paper's Regime B description calls the Gabor "spatially localized." Tracing the actual
tensor shapes: `GaborLayer`'s `in_features = T_in = 5`, `out_features = T_out`, and the
distance term `D(x)` is computed over the **last axis of the input**, which is the
**5-frame temporal history at a given pixel**, not spatial coordinates. The Gaussian gate
selects for *specific temporal trajectories* (e.g. "rapid recent intensification"), not
*specific spatial locations* — that job is already done by the DWT's spatial subband
decomposition. This is a discrepancy between the paper's wording and what the code
actually computes, worth correcting before submission.

### 1.4 Why the `√γ` coupling only holds at initialization

`self.linear.weight.data *= weight_scale * sqrt(gamma)` sets the *initial* scale of `W`,
but `W` is a free, trainable `nn.Parameter`. Gradients move it independently of `γ`
afterward. **Confirmed from the CIKM checkpoint:** `γ` grew 20–25× during training in the
HF bands, yet the *effective oscillation* (`λ·f·‖W‖`) actually **decreased**. If the `√γ`
coupling still held post-training, a 20× jump in `γ` should have driven oscillation up, not
down. It didn't — proof that `W` and `γ` decouple once training starts. This matters
directly for the regime experiment design (§5) and is why gamma needed to be frozen to
isolate `freq_multiplier` as a clean experimental axis.

---

## 2. GaborLayer code changes

Two changes were made to the original layer, for two independent reasons.

### 2.1 Change 1 — `self.gamma`: `nn.Parameter` → `register_buffer` (frozen)

**Why:** The regime-sweep experiment (§5–6) requires gamma to be fixed at a specific
value derived from a trained checkpoint, so that `freq_multiplier` is the only thing being
tested. `requires_grad_(False)` would work, but a **buffer** is structurally safer — it can
never receive a gradient, can never be picked up by `.parameters()` for an optimizer group,
and still saves/loads correctly via `state_dict()` and moves correctly with `.to(device)`.

### 2.2 Change 2 — bias re-scaled to match `weight_scale·√γ`

**The bug this fixes:** the original bias init, `Uniform(-π, π)`, is completely
independent of `gamma` and `weight_scale`. Since bias enters the sine argument exactly the
same way the signal does (`z = λ·f·(W·x + b)`), and `λ·f` multiplies *both* terms equally,
the **ratio** of bias to signal is fixed by their respective base scales — not by
`freq_multiplier`. This ratio was quantified for CIKM:

| Band | bias / signal ratio | consequence |
|---|---|---|
| LL | **206×** | bias completely swamps the input-driven term |
| HF | **28×** | bias still dominates, less severely |

No choice of `freq_multiplier` can fix this — `λ` scales both terms by the same factor, so
the ratio is architecturally locked in. At the very large `λ` values later shown to be
needed for the regime ladder, this bias, once multiplied by `λ`, wraps many multiples of
`2π` — effectively **randomizing each neuron's phase**, drowning out the input-driven
signal the calibration was trying to control.

**The fix:**
```python
# before (the bug):
self.linear.bias.data.uniform_(-np.pi, np.pi)

# after (matches the weight's own scaling exactly):
self.linear.bias.data = (2 * torch.rand(out_features) - 1) * weight_scale * torch.sqrt(self.gamma)
```
This makes `bias_std = weight_scale·√γ/√3`, which has the **identical functional form**
as `E[‖W‖]`. Whatever `γ`/`weight_scale` a dataset ends up with, bias and signal now scale
together automatically — no more per-dataset magic numbers needed to keep bias in check.

**Why RMS, not the ±π range endpoint, was used to quantify dominance:** the comparison
needs to be apples-to-apples — `E[‖W‖]` is a *typical* (expected) magnitude, so bias needs
its typical magnitude too, which is the standard deviation, `π/√3 ≈ 1.814`, not the
worst-case bound `π`. Verified this choice doesn't change the conclusion: using the
worst-case bound instead only changes the dominance ratio by `√3 ≈ 1.73×` (206× → 357× for
LL) — same order of magnitude, same conclusion either way.

### 2.3 Full corrected GaborLayer

```python
class GaborLayer(nn.Module):
    """
    Gabor activation with FROZEN gamma (bandwidth fixed at init, not learned)
    and gamma-matched bias scaling.
    """
    def __init__(self, in_features, out_features, weight_scale,
                 alpha=1.0, beta=1.0, freq_multiplier=1.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)

        gamma = torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        self.register_buffer('gamma', gamma)      # frozen: buffer, not Parameter

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
```

---

## 3. Calibrating `freq_multiplier`: the z-target ladder

### 3.1 Why `z` (the sine's argument scale) is the calibration target

`z = λ·f·(W·x)` is what actually determines whether `sin(z)` behaves linearly or
oscillates. The mean-field target used throughout:

```
z ≈ λ · f_mean · E[‖W‖],   E[‖W‖] ≈ weight_scale · √(γ / 3)
```

(`f_mean = 0.5` since `freq ~ Uniform(0,1)`; the `1/3` comes from the variance of a
Kaiming-uniform-initialized weight row.)

### 3.2 Why the ladder is `[0.1, 0.3, 0.8, 1.6, π]`

These aren't arbitrary — each is a physically meaningful point on `sin(z)` itself:

| z | sin(z) | deviation from linear | regime |
|---|---|---|---|
| 0.10 | 0.0998 | 0.2% | deep Regime A |
| 0.30 | 0.296 | 1.5% | still near-linear |
| 0.80 | 0.717 | 10.3% | transitional |
| 1.60 | 1.000 | 37.5% | sits at sin's peak (~π/2) |
| π | 0.000 | 100% | completes a half-cycle |

### 3.3 Why π, not π/2, is the top of the ladder

`sin(z)` is **monotonic** — no turning point — only on `[-π/2, π/2]`. A neuron whose
typical input range stays inside that interval behaves like a saturating S-curve (similar
to tanh): input goes up, output goes up, smoothly, no reversal. That is still
qualitatively Regime A, even though `z` reaches sin's numerical maximum at `π/2`.

**Genuine oscillation** requires the *spread* (`σ`) of `z` across the input distribution to
extend *past* the turning point, so the function goes up, comes back down, and (further
still) crosses back to negative — a non-monotonic response to input. The threshold for
this was made concrete:

| σ (z std) | typical (±1σ) range | periods traversed | behavior |
|---|---|---|---|
| π/4 | [-0.79, 0.79] | 0.25 | entirely inside the monotonic zone — pure S-curve |
| π/2 | [-1.57, 1.57] | 0.50 | exactly touches both turning points — boundary case |
| **π** | **[-3.14, 3.14]** | **1.00** | **one full period — genuine up-down-up oscillation** |
| 2π | [-6.28, 6.28] | 2.00 | two full periods within one typical input swing |
| 3π | [-9.42, 9.42] | 3.00 | high-frequency oscillator |

`σ = π` is the smallest value at which the *typical* input range completes a full cycle
rather than merely touching one extremum — that's why it anchors the top of the ladder,
not `π/2`.

### 3.4 Is there more variation beyond π?

Yes — 2π and 3π give 2 and 3 oscillation cycles within a typical input swing,
respectively. The ladder stops at one period for a practical reason, not a mathematical
one: as the number of cycles-per-typical-input-swing grows, the function becomes
increasingly rough (nearby inputs produce very different outputs), which is the documented
difficulty with high-frequency periodic activations in SIREN-style networks (Sitzmann et
al., NeurIPS 2020 — general finding, worth verifying the exact figures before citing).
Extending the ladder to 2π/3π as additional levels is a legitimate next step if the
one-period level doesn't already reveal the pattern of interest.

---

## 4. Why the originally-tried range (0.1–4.0) meant different things per dataset

### 4.1 The hidden role of `beta`/`weight_scale`

`z` depends on the **product** `λ · weight_scale · √γ`, not `λ` alone. Because CIKM used
`beta=100` (→ `γ_init=0.01`) and Shanghai used `beta=0.17` (→ `γ_init=5.88`) — a **588×**
gap — the same `freq_multiplier=4.0` landed in completely different places:

```
CIKM   (beta=100):    freq_multiplier=4.0  ->  z_HF = 0.029   (firmly linear)
Shanghai (beta=0.17): freq_multiplier=4.0  ->  z_HF = 2.80    (89% of the way to π)
```

### 4.2 Consequence

The 0.1–4.0 sweep wasn't testing a fixed "amount of sinusoidal-ness" across datasets — it
was riding on top of wildly different `beta`-driven baselines. For Shanghai's HF band it
reached genuine oscillation; for CIKM it structurally could not, no matter which value in
that range was chosen.

### 4.3 Did the sweep ever explore the full range of `sin`? — No, not for CIKM

`sin(0.029) = 0.030` — a 0.4% deviation from linear, even at the *top* of the tried range.
CIKM's Gabor never left the near-linear region across the entire 0.1–4.0 sweep. This
raises a real concern for the paper: the finding "CIKM favors near-linear Gabor" could be
a genuine atmospheric result, **or** it could be a structural artifact of choosing
`beta=100` for CIKM, which mathematically prevented the oscillatory regime from ever being
reached — regardless of what the data would have preferred if given the chance. This needs
at least one CIKM run reaching `z ~ 1–3` (matching Shanghai's HF reach) to distinguish the
two explanations.

---

## 5. Bugs found and fixed in `gabor_regime_calibrator.py`

Three real, quantifiable bugs were found in the final audit — not just style issues.

### 5.1 Jensen's inequality gap (the significant one)

`expected_W_norm` computed `weight_scale · √(E[γ])`, but the math actually needs
`weight_scale · E[√γ]`. These are **not equal** for a random variable — `sqrt` is concave,
so `E[√γ] ≤ √E[γ]` always (Jensen's inequality), with the gap growing with `γ`'s variance.
Since `alpha=1` makes `Gamma(1, β)` an **Exponential distribution** (high variance,
coefficient of variation = 1), the gap is not negligible.

**Exact correction**, using the Gamma-function moment formula
`E[X^p] = Γ(α+p)/Γ(α) · (1/β)^p`:

```
E[√γ] = Γ(α + 0.5) / Γ(α) · √(γ_mean / α)
```

At `α=1`: correction factor `= Γ(1.5)/Γ(1) = √π/2 ≈ 0.8862`. **The uncorrected formula
overestimated `E[‖W‖]` by ~12.8%, meaning every `freq_multiplier` value the calibrator
produced was ~11.4% too small.** Fixed by implementing the exact formula, generalized for
any `alpha`, not hardcoded to `alpha=1`.

### 5.2 `alpha` silently dropped before reaching the calibration formula

`--alpha` is a real, user-facing CLI parameter, correctly used when sampling `gamma` for
the actual `GaborLayer`. But it never reached `calibrate_freq_multiplier` /
`expected_W_norm` — those silently behaved as if `alpha=1` always. If `alpha` were ever
changed from the default, the design-target table and the empirical cross-check table
would have silently diverged for a reason unrelated to the actual experiment. Fixed by
threading `alpha` through both functions.

### 5.3 Confounded random seeds across the level sweep

`seed=seed + i` (where `i` is the level index, 0–4) meant **every level within the same
band got a different random draw** of `gamma`, `mu`, `W`-direction, and `freq` — not just a
different `freq_multiplier`. With only 10–20 neurons per layer, comparing L0 to L4 was
comparing *different λ* **and** *entirely different random neurons* at the same time.
Fixed by holding the seed **constant** across all 5 levels within a band — verified
afterward that `gamma` is bit-identical at every level, isolating `freq_multiplier` as the
sole varying factor.

### 5.4 `x_std` hardcoded and unreachable from the CLI

The empirical cross-check assumed `x ~ N(0, 1)` (`x_std=1.0`), which happens to roughly
match the description "typically [-1,1], up to [-3,3]" given for the real AE latents — but
this was never verified, and there was no way to override it without editing the source.
Fixed with a `--x_std` CLI flag, and, more importantly, a `--real_latents_path` option to
load **actual encoded AE latents** from a `.pt`/`.npy` file instead of synthetic Gaussian
noise — capturing the true shape (e.g. right-skew from non-negative reflectivity), not just
the variance, since a Gaussian with matching std still cannot reproduce skewed tails.

---

## 6. Design choice: freeze gamma only, keep `W` and `freq` trainable

**Why this is consistent with the hypothesis:** the experiment tests "does a
better-conditioned `γ` init improve the final result," not "is this exact frozen regime
optimal at convergence." Since the network is allowed to adapt `W` and `freq` from
wherever they start, letting them stay trainable is exactly the right setup for an
"init matters" framing — the network finds its own regime from a controlled starting
point. Freezing `freq` too would instead test a *locked* regime hypothesis, a different
(stronger, and more fragile) claim.

**Beta derivation for the frozen gamma:** rather than reusing a checkpoint's exact
per-neuron `gamma` tensor (which ties every grid cell to one specific training run's random
seed), the chosen approach recomputes `beta = alpha / mean(gamma_learned)` and samples a
**fresh** `Gamma(alpha, beta)` tensor centered at that mean, then freezes it. This preserves
the distributional character of the original init methodology while centering it correctly.

---

## 7. The empirical-vs-analytical gap, and the bias-in-z question

### 7.1 Why `z_target` uses `Wx` only, not `Wx + b`

Bias is a **per-neuron constant** — for a fixed neuron, it does not vary as input `x`
varies. The `z_target` ladder was built to answer "does this neuron's output change
nonlinearly *as the input trend changes*," which is a question about input-driven spread
only. Excluding bias here is conceptually correct for that specific question — bias
doesn't create spread across different inputs, it just shifts where a given neuron's
output sits.

### 7.2 But bias is not irrelevant — it explains part of the earlier "empirical exceeds
analytical" gap

The empirical cross-check (which runs the real `nn.Linear`, and therefore *does* include
bias automatically) showed realized `z_std` running 1.5–2× higher than the analytical
target. Originally this was attributed only to `gamma`'s and `freq`'s own per-neuron
variance. A follow-up decomposition (law of total variance, splitting `z`'s pooled
variance into within-neuron/input-driven and between-neuron/bias-driven parts) showed:

```
fraction of total pooled z variance coming from bias-driven between-neuron variation: ~43%
```

This is a real, previously under-attributed contributor — not a minor rounding effect.

### 7.3 Why this fraction doesn't change with `beta`

Bias and `‖W‖` now share the **identical** `weight_scale·√γ` coupling (§2.2's fix). Checked
numerically across a 250× range of `gamma_mean` (0.023 to 5.88): the ratio of bias's
typical magnitude to the signal's typical magnitude stayed fixed at **~0.886** every time.
Whatever `beta` a dataset is calibrated to, bias's relative contribution to the pooled
variance is architecturally fixed — it does not need separate re-derivation per dataset.

The calibrator's reading-guide text was updated to reflect this fuller, corrected
explanation of the gap (previously it mentioned only gamma/freq variance).

---

## 8. What oscillation actually means for rainfall prediction

### 8.1 The structure the Gabor operates on

`GaborLayer`'s `nn.Linear(T_in, T_out)` = `Linear(5, 10 or 20)`. Each output neuron
produces **one future timestep** from a learned linear combination of the **same 5 input
frames** — the layer performs `T_out` parallel, independent one-shot regressions from a
5-frame history, not a recurrently evolving state.

### 8.2 What linear (Regime A) vs oscillatory (Regime B) buys, physically

- **Linear:** `sin(z) ≈ z` — output is a linear function of the recent temporal trend.
  Equivalent to trend-extrapolation / persistence-with-trend. Correct for large-scale
  convective bulk that advects smoothly over the forecast horizon.
- **Oscillatory:** `sin(z)` is non-monotonic in the trend — the model can represent
  **"a stronger recent trend predicts an *upcoming reversal*,"** not only "the trend
  continues." This is the mechanism that could let HF content represent convective-cell
  life-cycle behavior (initiate → intensify → peak → dissipate) rather than assuming
  persistence of fine-scale structure.

### 8.3 An honest limitation of this story

Because each of the `T_out` future frames is an **independent** regression from the same 5
inputs, the sine's non-monotonicity operates on the *trend → single-frame-output* mapping,
not on the predicted time axis itself. It does not make frame T+8 oscillate relative to
T+7 — any smoothness or roughness across the predicted sequence comes from how the
different neurons' individually-learned `W` rows and `f` values relate to each other, not
from the sine literally cycling through time.

### 8.4 Reframed physical verification target

Given the above, the meteorological quantity that should predict the optimal oscillation
level per dataset is not directly "wind vs. updraft strength" but the **decorrelation /
life-cycle timescale of fine-scale features relative to the model's horizon** — how often
small structures are born and die within the forecast window. This is directly measurable
from the radar data itself (HF-band temporal decorrelation time), which is a cleaner,
more falsifiable claim than a general wind/updraft story, because it's measuring exactly
the quantity the 5→T_out Gabor filter actually has access to.

---

## 9. Supporting tools built alongside this work

| Tool | Purpose |
|---|---|
| `gabor_checkpoint_inspector.py` | Reads a trained checkpoint, compares learned `γ`/`freq` to their init distributions, and classifies each layer as INIT_DOMINATES / MIXED / LEARNED_DOMINATES to decide whether a regime sweep should freeze parameters or not. |
| `gabor_regime_calibrator.py` | Derives `beta` from a learned `gamma` mean, builds the 5-level `freq_multiplier` grid per band, and empirically cross-checks the realized `sin(θ)` statistics via an actual forward pass through the (now frozen-gamma, gamma-matched-bias) `GaborLayer`. |
| `gabor_sweep_matrix.py` | Parses training/eval logs (handling both CIKM's one-log-per-run layout and Shanghai's concatenated-log layout) to build the `(freq_low × freq_high)` score matrices for CSI-M, CSI-35, CSI-40, HSS, MSE, SSIM, PSNR, etc. |

A related infrastructure issue surfaced while reviewing the Shanghai sweep script: its
original `exp_note` tag was constant across all 20 off-diagonal runs, meaning every run's
checkpoint directory was the same path — each new run's `ckpt-best.pt` silently overwrote
the previous one. Scores in the log survived (log files append), but only the
last-finishing checkpoint per GPU remains on disk. This was fixed by switching to a
per-combination tag (`Shanghai_flow${F_LOW}_fhigh${F_HIGH}`), matching what the CIKM script
already did correctly.

---

## Appendix: Quick-reference formulas

```
E[γ]  for Gamma(alpha, beta) [rate parameterisation]      = alpha / beta
beta  (to hit a target learned-gamma mean)                = alpha / gamma_mean

E[‖W_row‖]  (corrected, Jensen-exact)
    = weight_scale * [Gamma(alpha+0.5)/Gamma(alpha)] * sqrt(gamma_mean/alpha) / sqrt(3)

z_target -> freq_multiplier
    freq_multiplier = z_target / (0.5 * E[‖W_row‖])

z ladder (physically anchored):  [0.10, 0.30, 0.80, 1.60, pi]
    representing: near-linear, mild, transitional, peak-curvature (~pi/2), half-cycle (~pi)

bias std (after the fix) = weight_scale * sqrt(gamma) / sqrt(3)   <- identical form to E[‖W‖]
bias-to-signal ratio (structural constant, independent of beta)  ~= 0.886
```
