# THE_GABOR — what was created, and why

Everything for these experiments lives inside `THE_GABOR/`. **No file outside
this directory was modified.** Existing repository code is imported and reused;
where behaviour had to change, the code was copied or wrapped inside
`THE_GABOR/` and the original left byte-identical (verified by sanity check 19,
which compares `git status` and the sha256 of 9 protected files against
`configs/repo_baseline.txt`).

One exception, recorded for transparency: on 2026-08-25 17:47 an accidental IDE
keystroke corrupted `models/DAWNCast/dawncast.py` (`x.shape` → `x.shapeIn`, a
syntax error at line 421). This was **not** part of this work. The file was
restored to its committed content (hash back to `340b8924…`, `git diff` clean)
and the broken copy preserved at `logs/_recovered/dawncast.py.broken_20260825_1747`.

---

## 1. Models

### `models/gabor_mlp_model.py` — the controlled model (Experiment 1)
A deliberately minimal network used to isolate the Gabor + MLP mechanism:

```
Input → Lifting → DWT → per-subband (Gabor + MLP → fusion) → IDWT → Projection → Output
```

Removed relative to full DAWN-Cast: SRST, STR/AFNO, spectral/spatial/Fourier
refinement, the Gabor residual bypass, and WGTM aggregation. `GaborLayer`,
`_ConvNormAct` and `TransformBlock` are copied verbatim from
`models/DAWNCast/dawncast.py` so the Gabor formulation studied is the real one.
`freq_multiplier = 1.0` on every subband and `freq ~ U(0,1)` — no regime prior.
Trains on FACL only (`utils.utilspp.RandomScheduling`, imported unmodified);
`predict()` returns the *same tensor object* for `facl_loss` and `total_loss`.

### `models/dawncast_transfer.py` — DAWN-Cast with per-subband Gabor parameters
**Why it exists:** the original `WGTMBlock` *derives* HF Gabor frequency
multipliers by interpolation, so an individual subband cannot be addressed or
initialised independently:

```python
freq_mid = (freq_multiplier_low + freq_multiplier_high) / 2
freq_i   = freq_multiplier_high * (1 - a) + freq_mid * a      # a = i/(level-1)
```

Here `weight_scale` / `alpha` / `beta` / `freq_multiplier` are instead supplied
**per subband** — scalar (broadcast), sequence, or dict keyed by subband name —
ordered `LL, HF_level_1 … HF_level_J`. Nothing is interpolated implicitly.

**The architecture is unchanged.** Every component not part of this parameter
plumbing (`GaborLayer`, `FATBlock`, `SRSTResBlock`, `STRModule`,
`TransformBlock`) is *imported* from the original file rather than copied.
Verified: identical 127-tensor `state_dict` and **0.0** output difference versus
the original when given equivalent parameters.

---

## 2. Data

### `datasets/sevir_regime_dataset.py` — RANDOM / STORM filtering
Every SEVIR `vil` row belongs to exactly one HDF5 family, encoded in
`file_name`: `SEVIR_VIL_RANDOMEVENTS_*` (16 483 rows) or
`SEVIR_VIL_STORMEVENTS_*` (3 910). `event_type` is an exact complement (NaN for
all RANDOM rows); `file_name` is used because it marks both regimes positively.

Filtering goes through the `catalog_filter` hook the existing loader already
exposes, composed with the default `pct_missing == 0`. `CATALOG.csv` and the
dataset code are untouched. Date boundaries are applied **before** the regime
mask, so event-level train/val/test separation is preserved and no event's
sequences cross a split. Every run asserts the opposite regime contributed zero
rows.

`pin_resize_antialias()` pins `antialias=True` on the 384→128 resize of the
pixel loader. torchvision changed this default between the two environments
(`earthformer` 0.22 → True; `earthformer_old` 0.13 → None/False), and switching
it changes the validation target by max 0.68 / mean 0.226 (~45% of signal mean)
— which would move every CSI number for reasons unrelated to the model. Applied
to the dataset *instance*, so `datasets/dataset_sevir.py` is never modified.

---

## 3. Utilities

| file | purpose |
|---|---|
| `utils/experiment.py` | shared training harness: FACL-only loop, Gabor logging, checkpointing, shared-init loading, `after_init_load()` hook |
| `utils/init_checkpoint.py` | the identical-initialisation guarantee (below) |
| `utils/gabor_probe.py` | fixed deterministic probe; `D(x)` computed inline so it works with both `GaborLayer` variants |
| `utils/gabor_logging.py` | W&B scalars, probe curves, `gabor_state` export, neuron-mean summaries |
| `utils/gabor_visualization.py` | per-neuron and neuron-mean figures |
| `utils/gabor_transfer.py` | donor → DAWN-Cast parameter transfer + freezing |
| `utils/metrics_per_threshold.py` | per-threshold CSI (CSI-181/219 etc.) on top of `utils/metrics.py`, which only returns the threshold average |
| `models/dawncast_ablations.py` | the ablation variants a–g (+ `f_no_srst`), one component removed each |

### Identical initialisation
For each `(model, space, seed)` **one** model is initialised and written to
`checkpoints/_initial/initial_<space>_<signature>_seed<N>.pt`; every run loads
*that same file*. Each run copies it to its own `initial_model.pt` and records
the sha256 in `initial_checkpoint.json`, so "byte-for-byte identical" is
verifiable from the runs alone. The signature includes the model class — added
after discovering that the controlled model and DAWN-Cast otherwise produced the
same filename and could collide.

### Transfer (`utils/gabor_transfer.py`)
Maps donor → recipient by component, shape-checking every tensor and failing
loudly rather than transferring partially:

```
net.block_ll.gabor.*      → dawncast.wgtm.fat_ll.gabor.*             (LL)
net.blocks_hf.{i}.gabor.* → dawncast.wgtm.fat_hf_streams.{i}.gabor.* (HF_level_{i+1})
net.block_ll.mlp.*        → dawncast.wgtm.fat_ll.mlp.*
net.lifting.*             → dawncast.lifting.*
net.projection.*          → dawncast.projection.*
```

Gabor tensors are channel-independent (they depend only on `t_in→t_out`), so a
subband transfers regardless of its channel width. The donor's **3** subbands
map 1:1 onto DAWN-Cast only with `hf_mode='separate'`; `'shared'` has 2 Gabor
modules and is rejected.

**`donor_freq_multipliers()` matters.** `freq_multiplier` is a plain float, not
a parameter, and the Gabor computes `sin(freq_multiplier · freq · linear(x))`.
The donors trained at `1.0` while DAWN-Cast defaults to `4.0`, so transferring
`freq` without carrying `freq_multiplier` would silently rescale every learned
frequency 4×. This is the main reason the per-subband interface was needed.

---

## 4. Runners and scripts

| file | purpose |
|---|---|
| `run_pixel.py` | controlled model, pixel SEVIR (`T_in=5, T_out=20`) |
| `run_latent.py` | controlled model, latent SEVIR (`sevir_lr_latent_32`) |
| `run_dawncast_transfer.py` | full DAWN-Cast on latent SEVIR, initialised from a donor; `--transfer` / `--freeze` / `--target_regime` / `--donor_regime` |
| `make_init.py` | creates the one shared initial checkpoint per (space, seed) |
| `sanity_check.py` | 19-point pre-training report |
| `eval_test.py` | test-set evaluation + Excel/CSV results table |
| `compare_regimes.py` | RANDOM-vs-STORM overlays, evolution and divergence plots; back-fills mean panels from saved `.npz` |
| `run_all_gpu1.sh` | Experiment-1 master script (sequential) |
| `run_batch2_gpu1_88.sh` | batch-2 transfer runs (sequential, GPU 1 of `.88`) |

---

## 5. Gabor measurement

Three quantities are logged and never conflated:

- **A. raw sinusoid** `sin(z)` — a *curve* over a fixed probe
- **B. frequency** `freq` and `effective_frequency = freq_multiplier · freq` — scalar mean/std/min/max
- **C. complete response** `sin(z) · exp(-0.5·D·gamma)` — a separate curve

The probe is built from constants only (`s = linspace(-3,3,201)`,
`u = ones(T_in)/√T_in`, `x_probe = s·u`) so it is identical across regimes,
spaces, epochs and seeds. Probed neurons are evenly spaced indices — and because
the Gabor maps `t_in → t_out`, **neuron `n` is predicted frame `n`**, i.e. the
neuron axis is forecast lead time (verified: a spike injected at Gabor output
`j` moves only output frame `j`, zero leakage).

Neuron-mean panels show mean, ±1 std *and* RMS, because neurons can sit in
opposite phase — measured cancellation ranges from ~0% (HF_level_1) to ~72%
(HF_level_2), so a mean-only plot would misread.

---

## 6. Known caveats

- **Step parity is not achieved between regimes.** `--limit_train_batches 2000`
  caps steps/epoch but does not equalise the arms: at `seq_len=25, bs=4` RANDOM
  has 4 710 batches/epoch and STORM only 1 242, so the cap binds for RANDOM
  (→2 000) and never for STORM (→1 242). Totals over 50 epochs: RANDOM 100 000
  steps, STORM 62 100. **Read RANDOM-vs-STORM trajectories on the step axis.**
- **Donor training lengths differ**: storm donor best at step 49 680, random
  donor at 80 000 (both epoch 40). Exp 1 vs Exp 2 therefore differ in donor
  regime *and* donor training length.
- **No default-init baseline was run**, so the transfer experiments show
  matched-vs-mismatched but cannot show either beats ordinary initialisation.
- **The fusion Conv3d is never frozen.** In the frozen runs the Gabor and MLP
  streams are fixed but the 1×1×1 conv combining them stays trainable, so the
  model can still reweight between the two frozen streams.
- Donor Gabors were learned in a model with **no SRST/STR** beside them; in
  DAWN-Cast they sit alongside spectral refinement.
- Frozen parameters are a small fraction of DAWN-Cast: 269 028 of 59.5 M
  (0.45%), since the model is dominated by SRST/AFNO.
