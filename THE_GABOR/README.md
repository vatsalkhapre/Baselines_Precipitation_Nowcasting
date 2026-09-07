# THE_GABOR — Experiment 1

**Question measured here (and only this):**

> Does the Gabor component learn differently when trained on different
> precipitation regimes?

Concretely: two models start from the **exact same initial checkpoint**, one is
trained on SEVIR **RANDOM** events, the other on SEVIR **STORM** events, and the
evolution of the Gabor operator is recorded throughout training.

Not implemented and not run here: Gabor initialisation transfer, parameter
freezing, cross-regime initialisation, matched-statistics initialisation,
climate-variable analysis. The code is structured so those can be added later
(`gabor_state.pt` is saved by every run), but nothing of the kind happens now.

No file outside `THE_GABOR/` is modified. Existing code is imported and reused;
where code had to change it was copied into `THE_GABOR/` first.

---

## 1. Controlled architecture

```
Input (B, T_in, C, H, W)
  -> Lifting            (frame-wise, C -> hidden_dim)
  -> spatial DWT        (J-level)
  -> per wavelet subband, independently:  Gabor + MLP -> simple fusion
  -> IDWT
  -> Projection         (frame-wise, hidden_dim -> C)
Output (B, T_out, C, H, W)
```

Retained: Lifting, DWT, Gabor, MLP, Gabor+MLP fusion, IDWT, Projection.

Removed w.r.t. full DAWN-Cast: SRST, STR, AFNO, spectral refinement, spatial
refinement, Fourier refinement, the Gabor residual bypass around SRST, and the
WGTM aggregation logic. Nothing was added beyond what the data flow requires.

`GaborLayer`, `_ConvNormAct` and `TransformBlock` are copied verbatim from
`models/DAWNCast/dawncast.py`; the Gabor formulation is unchanged:

```
G(x) = sin(freq_multiplier * freq * linear(x)) * exp(-0.5 * D(x) * gamma)
D(x) = ||x||^2 + ||mu||^2 - 2 x mu^T
```

### Gabor initialisation — no regime bias

* `freq_multiplier = 1.0` for **every** subband (LL and all HF levels).
  The DAWN-Cast low/high frequency priors are not used.
* `freq ~ Uniform(0, 1)` (`torch.rand`), as in the original layer.
* `mu`, `gamma`, `linear.weight`, `linear.bias`: the existing stochastic
  initialisation, with a single `weight_scale`/`alpha`/`beta` shared by all
  subbands.

### Wavelet configuration (identical for RANDOM and STORM)

| setting | default | flag |
|---|---|---|
| `wave` | `db6` | `--wave` |
| `wavelet_level` | `2` | `--wavelet_level` |
| `hf_mode` | `separate` | `--hf_mode` |

`J=2` + `separate` gives three independently-tracked Gabor subbands —
`LL`, `HF_level_1`, `HF_level_2` — which is what makes "which subband changed
most" answerable. `J=1` reproduces the DAWN-Cast pixel default.

---

## 2. Loss — FACL only

`total_loss = FACL(prediction, target)`, using the existing implementation
`utils.utilspp.RandomScheduling` (the loss DAWN-Cast itself trains with — see
`DAWNCastForecaster.predict` in `models/DAWNCast/dawncast.py`). It is imported,
not copied or modified.

`predict()` returns the *same tensor object* under both `facl_loss` and
`total_loss`, and the training step asserts
`total is facl` **and** `torch.equal(total, facl)` on every step. No MSE, L1,
perceptual, SSIM, auxiliary or reconstruction term exists anywhere in the graph.

W&B keys: `loss/facl`, `loss/total`, plus `loss/facl_minus_total` (always 0).

---

## 3. RANDOM vs STORM

Every `vil` row of the SEVIR catalog belongs to exactly one HDF5 family,
encoded in `file_name`:

```
vil/2017/SEVIR_VIL_RANDOMEVENTS_....h5   -> RANDOM   (16483 rows)
vil/2017/SEVIR_VIL_STORMEVENTS_....h5    -> STORM    ( 3910 rows)
```

`event_type` is an exact complement of this split (NaN for all RANDOM rows, a
storm label for all STORM rows). `file_name` is used because it marks both
regimes positively rather than by a null test.

Filtering is done through the `catalog_filter` hook the existing loader already
exposes (`THE_GABOR/datasets/sevir_regime_dataset.py::RegimeCatalogFilter`),
composed with the default `pct_missing == 0` mask. `CATALOG.csv` and the
existing dataset code are untouched. Every run asserts, from the filtered
catalog the loader actually holds, that the other regime contributed zero rows.

### Splits (unchanged from `datasets/get_datasets.py`)

Date boundaries are applied **before** the regime mask, so event-level
separation is preserved and no event's sequences cross a split:

| split | dates |
|---|---|
| train | `time_utc <= 2019-01-01` |
| val   | `2019-01-01 < time_utc <= 2019-06-01` |
| test  | `2019-06-01 < time_utc <= 2019-12-31` |

Events kept after `pct_missing == 0` and the duplicate-id fix:

| | train | val | test |
|---|---|---|---|
| pixel RANDOM | 9421 | 2596 | 3426 |
| pixel STORM  | 2485 |  424 |  627 |
| latent RANDOM| 9420 | 2596 | 3426 |
| latent STORM | 2485 |  424 |  627 |

Counts are printed and logged to W&B (`data/<split>_num_events`,
`data/<split>_num_sequences`, `data/<split>_num_batches`).

**Note on step counts — read trajectories on the STEP axis.** RANDOM has ~3.8x
more training events than STORM, so equal epochs do not mean equal gradient
updates. `--limit_train_batches 2000` caps steps/epoch, but at `seq_len=25`,
`batch_size=4` the two regimes have very different amounts of data available:

| | batches/epoch available | steps/epoch actually run | total over 50 epochs |
|---|---|---|---|
| RANDOM | 4710 | 2000 (cap binds) | 100,000 |
| STORM  | 1242 | 1242 (cap never binds) | 62,100 |

So the cap **does not** equalise the regimes — STORM simply runs out of data
first. Both arms still complete a full cosine LR cycle and a full FACL schedule
over their own `total_steps`, so each is internally consistent, but the epoch
axis is not comparable between regimes: epoch *e* is 2000*e* steps for RANDOM
and 1242*e* for STORM. Compare RANDOM against STORM on the **step axis**.

The latent runs use the same `--limit_train_batches 2000` and the same
`seq_len=25`, so each latent regime gets exactly the same steps/epoch as its
pixel counterpart (RANDOM 2000, STORM 1242). The pixel-vs-latent comparison is
therefore matched per regime, and the regime step gap is identical in both
spaces rather than differing between them.

### Latent regime metadata — case that applies

**Available.** The latent dataset ships its own `CATALOG.csv` that preserves the
original `file_name` column (`vil_latent/2017/SEVIR_VIL_STORMEVENTS_....h5`), so
RANDOM/STORM membership is recoverable with exactly the same mask. Latent regime
filtering is therefore supported and `run_all_gpu1.sh` runs latent RANDOM and
latent STORM. `--regime all` runs the standard unfiltered latent experiment.

The latent horizon matches pixel exactly (`T_in=5`, `T_out=20`, `seq_len=25`);
the latent HDF5 files hold 49 frames per event, so this is fully available. The
pixel and latent arms therefore differ **only** in the space they operate in.

The latent data is already encoded — the autoencoder is **not** run inside the
model. It is loaded only to decode predictions back to pixel space for
validation, exactly as the existing latent runner does.

---

## 4. Identical initialisation

For each `(space, seed)`:

1. `make_init.py` initialises **one** model and writes
   `checkpoints/_initial/initial_<space>_<signature>_seed<seed>.pt`.
2. The RANDOM run loads that file (`strict`).
3. The STORM run loads *that same file*.

Each run copies it to `<run>/checkpoints/initial_model.pt` and records its
sha256 in `<run>/initial_checkpoint.json`, printed at startup and logged to
W&B — so "byte-for-byte identical" is verifiable from the two runs alone.
`<signature>` is a hash of the architecture config, so a configuration change
cannot silently reuse an incompatible checkpoint.

---

## 5. The three Gabor quantities (kept strictly separate)

**A. Raw sinusoid** — `sin(z)`, `z = freq_multiplier * freq * linear(x)`,
logged as a **curve** over the fixed probe range:
`gabor/<subband>/sinusoid/response`

**B. Sinusoid frequency** — the learned scalar parameters:
`gabor/<subband>/freq/{mean,std,min,max}` and
`gabor/<subband>/effective_frequency/{mean,std,min,max}`
(`effective_frequency = freq_multiplier * freq`; both logged explicitly even
though `freq_multiplier = 1.0`), plus a per-neuron bar chart at
`gabor/<subband>/frequency/selected_neurons`.

**C. Complete Gabor response** — `sin(z) * exp(-0.5 * D(x) * gamma)`, logged as
a separate **curve** on the same probe and the same neurons:
`gabor/<subband>/gabor_response`
(the Gaussian factor alone is also logged: `gabor/<subband>/envelope_response`).

Also tracked per subband, as mean/std/min/max scalars:
`freq`, `effective_frequency`, `gamma`, `mu`, `linear_weight`, `linear_bias`;
plus periodic `wandb.Histogram` of `freq` and `gamma` only.

### What a Gabor "neuron" is — forecast lead time

`GaborLayer(t_in, t_out)` maps the *input* temporal axis to the *output*
temporal axis, so its output neuron `n` produces **predicted frame `n`**
(0-based). Verified by injecting a spike at one Gabor output index and
confirming only that output frame moves, with zero leakage — nothing after the
Gabor mixes time (the fusion is a `1x1x1 Conv3d`, the IDWT is spatial, the
projection is frame-wise).

* pixel (`T_out=20`), probed neurons `[0, 6, 13, 19]` -> predicted frames 1, 7, 14, 20
* latent (`T_out=10`), probed neurons `[0, 3, 6, 9]` -> predicted frames 1, 4, 7, 10

So the neuron axis of every per-neuron plot **is forecast lead time**, and the
frequency bar chart reads as "learned frequency vs lead time". Plot labels say
so explicitly.

Two things these curves are not: the Gabor is a purely *temporal* operator
applied identically at every channel and spatial position of its subband
(nothing plotted is spatial), and the x-axis `s` is the probe sweep
`x_probe = s * u`, `u = ones(T_in)/sqrt(T_in)` — i.e. all input frames set to
the same value. A curve reads "as the input sequence uniformly scales from -3
to +3, how does this subband's contribution to predicted frame n respond?"

### Neuron-mean panels (easier side-by-side reading)

Alongside the per-neuron traces, each probe checkpoint also logs a
neuron-averaged panel per subband — one line instead of K overlapping ones:

`gabor/<sub>/gabor_response/mean`, `gabor/<sub>/sinusoid/mean`,
`gabor/<sub>/envelope/mean`

Each panel shows **mean across neurons** (solid), **+/- 1 std** (band) and
**RMS** (dashed). The RMS is there because neurons of a subband can sit in
opposite phase, in which case the plain mean collapses toward zero while the
actual response magnitude does not — measured cancellation ranges from ~0%
(HF_level_1) to ~72% (HF_level_2), so showing the mean alone would misread.

Because W&B overlays *scalars* across runs but not *images*, the same
checkpoints also log scalar summaries that chart RANDOM against STORM directly:

```
gabor/<sub>/gabor_response/{mean_abs, rms, peak_abs}
gabor/<sub>/gabor_response/{neuron_mean_abs, neuron_std, phase_alignment}
gabor/<sub>/sinusoid/{mean_abs, rms, peak_abs}
gabor/<sub>/envelope/{mean_abs, rms, peak_abs}
```

`phase_alignment = neuron_mean_abs / mean_abs` is 1.0 when a subband's neurons
agree in phase and 0 when they cancel.

### Comparing two runs — `compare_regimes.py`

Reads the probe `.npz` files both runs already wrote and draws them on shared
axes (which W&B cannot do for images). It also **back-fills** the neuron-mean
panels for runs trained before those panels existed — the `.npz` files contain
everything needed, so no retraining is required.

```bash
# back-fill mean panels for a finished run
python -m THE_GABOR.compare_regimes --runs Gabor_pixel_SEVIR_random_seed0 --backfill

# RANDOM vs STORM overlays + divergence, all pushed to one W&B run
python -m THE_GABOR.compare_regimes \
    --runs Gabor_pixel_SEVIR_random_seed0 Gabor_pixel_SEVIR_storm_seed0 \
    --labels random storm --backfill \
    --wandb_run Gabor_pixel_SEVIR_compare_seed0
```

Produces, under `logs/_compare/<labels>/`:

* `<tag>/` — both regimes overlaid on one axes, per subband per quantity
* `evolution/` — each scalar summary vs epoch, both regimes on one axes
* `divergence/` — RMS distance between the two regimes' mean and RMS curves vs
  epoch. Both arms start from the same initial checkpoint, so this begins at 0
  by construction; the shape of the rise is the measurement.

### The fixed deterministic probe

Built from constants — no RNG, no training data:

```
s          = linspace(-3.0, 3.0, 201)          # plot x-axis
u          = ones(T_in) / sqrt(T_in)           # fixed unit direction
x_probe[p] = s[p] * u                          # (201, T_in)
```

Identical across RANDOM/STORM, pixel/latent, every epoch and every seed
(verified by hash in the sanity check). Probed neurons are evenly spaced
indices over the output dimension — deterministic, and the same for both
regimes.

Curves are logged at `init`, every `--gabor_probe_every_epochs` epochs, and at
`final`; PNGs are written to `logs/<run>/gabor_plots/<tag>/` and the raw arrays
to `logs/<run>/gabor_probe/gabor_probe_<tag>.npz` for post-hoc analysis.

---

## 6. Checkpoints

`checkpoints/<run_name>/checkpoints/`:

| file | contents |
|---|---|
| `initial_model.pt` | the shared initial checkpoint (identical across regimes) |
| `best_model.pt` | best validation CSI |
| `last_model.pt` | end of the most recent epoch |
| `final_model.pt` | end of training |
| `gabor_state_init.pt`, `gabor_state_best.pt`, `gabor_state.pt` | all Gabor parameters organised by wavelet subband |

`gabor_state*.pt` exists for future experiments only; nothing is transferred or
frozen here.

---

## 7. Running

```bash
# sanity checks only
python -m THE_GABOR.sanity_check              # add --skip_data to skip SEVIR I/O

# everything, one run at a time on GPU 1
bash THE_GABOR/run_all_gpu1.sh

# multiple seeds
SEEDS="0 1 2" bash THE_GABOR/run_all_gpu1.sh

# a single run by hand
python -m THE_GABOR.make_init  --space pixel --seed 0
python -m THE_GABOR.run_pixel  --regime random --seed 0
python -m THE_GABOR.run_pixel  --regime storm  --seed 0
```

W&B run names: `Gabor_pixel_SEVIR_{random,storm}_seed<N>`,
`Gabor_latent_SEVIR_{random,storm}_seed<N>`. Config carries `model`, `dataset`,
`space`, `regime`, `seed`, `wavelet`, `wavelet_level`, `hidden_dim`,
`frames_in`, `frames_out`, `freq_multiplier`, `FACL_only=true`.

---

## 8. Paths used

Resolved from the existing repository rather than invented:

| what | where |
|---|---|
| DAWN-Cast reference model | `models/DAWNCast/dawncast.py` |
| pixel runner reference | `run_alphapre_convlstm.py` |
| latent runner reference | `run_alphapre_convlstm_sevir_lr_latent.py` |
| SEVIR root / catalog | `datasets/get_datasets.py::DATAPATH['sevir']` + `CATALOG.csv` |
| SEVIR loader | `datasets/dataset_sevir.py` |
| latent SEVIR root / loader | `DATAPATH['sevir_lr_latent_32']`, `datasets/dataset_sevir_lr_latent.py` |
| FACL | `utils/utilspp.py::RandomScheduling` |
| metrics | `utils/metrics_valid.py`, `utils/metrics.py` |
| AE checkpoint | `Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth` |
| output root | `THE_GABOR/checkpoints/` (`--output_root`) |
| log root | `THE_GABOR/logs/` (`--log_root`) |

## 9. Directory layout

```
THE_GABOR/
    run_pixel.py            pixel SEVIR runner
    run_latent.py           latent SEVIR runner
    make_init.py            creates the ONE shared initial checkpoint
    sanity_check.py         19-point pre-training report
    run_all_gpu1.sh         master script, GPU 1, sequential
    models/gabor_mlp_model.py
    datasets/sevir_regime_dataset.py
    utils/experiment.py         shared training harness
    utils/gabor_probe.py        fixed deterministic probe
    utils/gabor_logging.py      W&B Gabor logging
    utils/gabor_visualization.py
    utils/init_checkpoint.py    identical-initialisation guarantee
    configs/repo_baseline.txt   working-tree baseline for sanity check 19
    checkpoints/  logs/
```
