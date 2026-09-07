# RESUMPTION PROTOCOL — READ THIS FIRST, EVERY SESSION

This is a multi-day, multi-server task. Your session will be interrupted by usage
limits one or more times before it's done. Follow this protocol so work continues
without me having to re-explain anything.

## State file (you own this)
Maintain `/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/RUN_STATE.md`.
Update it immediately after every one of these events — not batched, not at the end:
- a training/ablation job is launched (server, GPU, PID/session name, log path, start time)
- a job finishes (result location, key metric, compare-to-baseline verdict)
- a comparison/selection decision is made (which checkpoint won and why)
- results_table.tex or the ablation .tex is edited
- any experiment is skipped, blocked, or fails (and why)

Format: one checklist line per experiment item below (Part A steps, each Part B
ablation × dataset, each Part C ablation, items 4/5/6 sub-items), `[ ]`/`[x]`, plus
a running log section underneath with timestamped entries.

## On every session start (including the very first message of a new session)
1. Read `RUN_STATE.md` in full before doing anything else.
2. Do NOT trust "last known status" blindly — a job may have kept training or may
   have died during the gap. For every job marked in-progress, SSH in and check
   the actual process (tmux/screen session or PID) and tail the log file to get
   real status before deciding what to do next.
3. Reconcile: mark anything actually finished as finished, restart anything that
   died, leave anything genuinely still running alone.
4. Only then continue with the next unfinished step in the checklist, in the order
   given below (Part A → Part B in parallel → Part C in sequence → items 4-6).

## Launch discipline (so jobs survive a Claude Code disconnect)
Every training run must be launched detached — `tmux`/`screen` session or
`nohup ... &` with output redirected to a log file — never in the foreground of a
shell Claude Code is attached to. The GPU job's lifetime must not depend on
Claude Code's session staying alive. Name each tmux/screen session predictably
(e.g. `cikm_partA`, `sevir_ablation_c`) and record that name in RUN_STATE.md so a
future session can reattach/check it by name instead of guessing.

## Ambiguity when unattended
I will not always be available to answer questions mid-run. Before starting,
list every doubt/ambiguity you have about this prompt and ask me now, in this
session. Once work is underway and I'm not responding: do not block waiting for
an answer. Pick the most reasonable default, note the assumption and your
reasoning in RUN_STATE.md (with a `[ASSUMPTION]` tag), and keep going. If a
choice is high-stakes or hard to reverse (e.g. overwriting a checkpoint,
deleting data, picking which model "wins" a comparison that updates the .tex
table), still make the call, but flag it clearly in RUN_STATE.md so I can
review and correct it later rather than have it silently stall the pipeline.

## Deadline awareness
NOTE1 below (Sept 7 2026) applies to Part A only. If a session resumes and Part A
is still incomplete with the deadline close, prioritize finishing Part A over
starting new Part B/C work, and say so in RUN_STATE.md.

---

# TWO-STAGE TRAINING — DEFINITION (read before Part A)

> **SCOPE: every run in this document is PIXEL SPACE. No latent-space training or
> evaluation is to be performed.** (Same as NOTE2 in the ORIGINAL TASK below,
> restated here because it appears further down the file.) Latent-space work is
> referred to in a few places purely as *completed prior work* — for its tooling,
> or for results already obtained. Those references are never instructions to run
> anything: `sevir_lr_latent_32`, `run_latent.py` and the latent
> `run_dawncast_transfer.py` are all out of scope here.

"2 stage training" in this document means the following, and nothing else. A
fresh session with no chat history should be able to execute Part A from this
section alone.

## Stage 1 — temporal path only (no refinement)

Train a reduced model that contains ONLY the lifting, the wavelet temporal
path, and the projection:

```
Input (B, T_in, C, H, W)
  -> Lifting            (frame-wise, C -> hidden_dim)
  -> DWT                (J-level, per frame)
  -> per wavelet subband, independently:  Gabor + MLP -> fusion (concat + 1x1x1 Conv3d)
  -> IDWT
  -> Projection         (frame-wise, hidden_dim -> C)
Output (B, T_out, C, H, W)
```

- **Model:** `THE_GABOR/models/gabor_mlp_model.py` (`GaborMLPControlled`, factory `get_model`).
- **Runner (pixel):** `THE_GABOR/run_pixel.py`.
- **Trains:** lifting, projection, per-subband Gabor, per-subband MLP, fusion Conv3d.
- **Excludes:** SRST block, STRModule/AFNO, spectral + spatial refinement, and the
  Gabor residual bypass.
- **Gabor initialisation (deliberately neutral, no regime/level prior):**
  `freq_multiplier = 1.0` for EVERY subband, `freq ~ Uniform(0,1)`, and a single
  `weight_scale` / `alpha` / `beta` shared by all subbands (defaults 0.1 / 1.0 / 1.0).
  The DAWN-Cast low/high split and the per-level frequency interpolation are NOT used here.
- **Loss:** FACL only — `utils.utilspp.RandomScheduling`, imported unmodified.
  `predict()` returns the same tensor object for `facl_loss` and `total_loss`; no
  MSE/L1/SSIM/auxiliary term is added anywhere.
- **What Stage 2 consumes:** the **best-validation** checkpoint of the Stage-1 run —
  `<run>/checkpoints/best_model.pt` together with `<run>/checkpoints/gabor_state_best.pt`
  (both are written at the same step; the loader asserts this).

## Stage 2 — full DAWN-Cast, Gabor initialised from Stage 1

Train the complete DAWN-Cast model, but instead of random Gabor parameters,
initialise each subband's Gabor from the Stage-1 model trained on the SAME dataset.

- **Model:** `THE_GABOR/models/dawncast_transfer.py` (`DAWNCastPerSubband`).
  This is the FULL DAWN-Cast — architecture verified identical to
  `models/DAWNCast/dawncast.py` (same 127-tensor `state_dict`, 0.0 output difference).
  Unchanged components are imported from the original file, which is never modified.
- **Why a separate file is needed:** the original `WGTMBlock` *derives* the HF Gabor
  frequency multipliers by interpolating between `freq_multiplier_low/high`, so an
  individual subband cannot be addressed or initialised. The copy accepts
  `weight_scale` / `alpha` / `beta` / `freq_multiplier` **per subband** — scalar
  (broadcast), sequence, or dict keyed by subband name — ordered
  `LL, HF_level_1 ... HF_level_J`. Nothing is interpolated implicitly.
- **Transfer utility:** `THE_GABOR/utils/gabor_transfer.py`. Name mapping:

  ```
  net.block_ll.gabor.*        -> dawncast.wgtm.fat_ll.gabor.*                 (LL)
  net.blocks_hf.{i}.gabor.*   -> dawncast.wgtm.fat_hf_streams.{i}.gabor.*     (HF_level_{i+1})
  net.block_ll.mlp.*          -> dawncast.wgtm.fat_ll.mlp.*
  net.lifting.*               -> dawncast.lifting.*
  net.projection.*            -> dawncast.projection.*
  ```

  Every tensor is shape-checked and the transfer fails loudly rather than partially.
  After copying, each transferred tensor is verified equal to the donor.
- **Part A transfers the Gabor only** (`--transfer gabor`). Nothing is frozen in Part A.

### Hard requirements for Stage 2 (getting these wrong silently corrupts the result)

1. **`hf_mode` must be `separate`.** A Stage-1 donor has `1 + J` Gabor subbands
   (`LL, HF_level_1..J`); `hf_mode='shared'` gives DAWN-Cast only 2 Gabor modules and
   cannot map 1:1. The transfer code rejects `shared`.
2. **Carry the donor's `freq_multiplier` into Stage 2.** `freq_multiplier` is a plain
   float attribute, not a learnable parameter, and the Gabor computes
   `sin(freq_multiplier * freq * linear(x))`. Stage 1 trains at `1.0` while DAWN-Cast
   defaults to `4.0`, so transferring `freq` without also setting `freq_multiplier=1.0`
   silently rescales every learned frequency 4x. Use
   `gabor_transfer.donor_freq_multipliers()` and pass the result as the per-subband
   `freq_multiplier`. **This is the main reason the per-subband interface exists.**
3. **These must match between Stage 1 and Stage 2**, or the donor will not map:
   `wave`, `wavelet_level`, `hf_mode` (= `separate`), `hidden_dim`, `T_in`, `T_out`,
   `img_channel`, `img_size`.
4. Gabor tensors depend only on `(T_in, T_out)` — `linear.weight (T_out,T_in)`,
   `linear.bias (T_out,)`, `mu (T_out,T_in)`, `gamma (T_out,)`, `freq (T_out,)` — and are
   independent of a subband's channel width, so LL and HF transfer with identical shapes.
5. **Loss is FACL only in both stages.**

### The `freq_multiplier_*` values listed for Part B are NOT for Part A Stage 2

The `freq_multiplier_high / freq_multiplier_low` (and `weight_scale_high/low`) values
given in Part B are ordinary DAWN-Cast initialisation hyper-parameters for the
**ablation** runs, which do not use a Stage-1 donor. In Part A Stage 2 the Gabor comes
from the donor and `freq_multiplier` is `1.0` per subband (rule 2 above).

## Shared-initialisation discipline (for any set of runs that must be compared)

Initialise ONE model, save it, and have every run in the comparison load that exact
file — do not rely on re-seeding. `THE_GABOR/utils/init_checkpoint.py` does this:
checkpoints live in `THE_GABOR/checkpoints/_initial/initial_<space>_<signature>_seed<N>.pt`,
the signature hashes the architecture config *including the model class*, and every run
copies the file to its own `initial_model.pt` and records the sha256 in
`initial_checkpoint.json` so identity is verifiable after the fact.

## Per-dataset forecast horizons

| dataset | T_in | T_out |
|---|---|---|
| CIKM | 5 | 10 |
| SEVIR / Meteonet / Shanghai | 5 | 20 |

Ablation (c) ("MLP with 5 inputs, 10 outputs") is the CIKM case; in general the
replacement MLP is `T_in -> T_out` for the dataset being run.

## Existing tooling, and one gap that must be filled first

| purpose | file | status |
|---|---|---|
| Stage 1 (pixel) | `THE_GABOR/run_pixel.py` | exists — **use this** |
| Stage 2 (latent) | `THE_GABOR/run_dawncast_transfer.py` | exists — **out of scope, reference only** |
| **Stage 2 (pixel)** | — | **DOES NOT EXIST — must be written** |
| per-subband DAWN-Cast | `THE_GABOR/models/dawncast_transfer.py` | exists |
| transfer / freeze | `THE_GABOR/utils/gabor_transfer.py` | exists |
| test eval + xlsx/csv | `THE_GABOR/eval_test.py` | exists |
| LaTeX table builder | `THE_GABOR/make_latex_table.py` | exists |
| pre-flight checks | `THE_GABOR/sanity_check.py` | exists |
| what was built and why | `THE_GABOR/CHANGELOG.md` | exists |

`run_dawncast_transfer.py` is **latent-only**: it subclasses `LatentGaborExperiment` and
hardcodes `img_size=32, img_channel=4, T_out=20`. Part A Stage 2 is pixel-space, so a
pixel counterpart must be written first — subclass the pixel experiment from
`run_pixel.py`, reuse the same `after_init_load()` transfer hook, and keep the
`--transfer` / `--freeze` / `--donor_regime` / `--target_regime` flags.

## Caveats to carry into the writeup

- **Stage 1 is not exactly "DAWN-Cast without the refinement block."** `GaborMLPControlled`
  also drops the Gabor residual bypass that DAWN-Cast adds after SRST
  (`x = x_srst + gabor_residual`). Ablation (f) takes its numbers from Stage 1, so it is
  really "w/o SRST **and** w/o the Gabor residual path". If a strict "SRST removed only"
  variant is wanted, it needs a separate model that keeps the residual path.
- `--limit_train_batches` caps optimiser steps per epoch but does **not** equalise
  datasets of different sizes: if a dataset has fewer batches than the cap, the cap never
  binds and that arm simply trains for fewer steps. Check actual steps/epoch per run and
  compare trajectories on the step axis, not the epoch axis.
- Metrics can disagree. In the latent study the matched-regime donor won on the CSI
  family and HSS but lost on SSIM/PSNR/MSE. Decide and state the selection metric
  *before* declaring a winner for the .tex update (CSI is the metric used for
  `best_model.pt` selection).

---

# RESOLVED DETAILS — everything else needed to run this unattended

Settled after review; treat these as binding. Where a value was inferred from an
existing script or checkpoint, the source is named so it can be re-checked.

## Datasets covered by Part A

Part A (Stage 1 + Stage 2) runs on **all four** pixel-space datasets:
**CIKM, SEVIR, MeteoNet, Shanghai** — one Stage-1 run and one Stage-2 run each,
on the servers listed under "Running status".

## Training hyper-parameters (both stages, all datasets)

Taken from `scripts/scripts_run/run_models_dawncast_pixel_space.sh`, which is the
reference pixel-space DAWN-Cast recipe:

| setting | value |
|---|---|
| `EPOCHS` | 50 |
| `SEED` | 0 (Part A); see item E for extra seeds |
| `IMG_SIZE` | 128 |
| `IMG_CHANNEL` | 1 |
| `FRAMES_IN` | 5 |
| `FRAMES_OUT` | 20 (CIKM: **10**) |
| `SEQ_LEN` | 25 (CIKM: **15**) |
| `BATCH_SIZE` | 4 (SEVIR on .205: 8 total = 4/GPU across 2 GPUs) |
| `NUM_WORKERS` | 8 |
| `HIDDEN_DIM` | 64 |
| `SIZE_FACTOR` | 1.0 |
| `HF_MODE` | `separate` (required for Stage-1 -> Stage-2 mapping) |
| optimiser | AdamW, `lr=1e-4`, betas (0.90, 0.95), wd 1e-5, cosine schedule w/ 20% warmup |
| loss | FACL only, `facl_const_ratio=0.1` |
| conda env | **`earthformer`** on every server |

`SEQ_LEN = FRAMES_IN + FRAMES_OUT`. The per-dataset `wave`, `wavelet_level`,
`sparsity_threshold`, `spectral_blocks`, `spectral_hidden_size_factor` and
`k_spatial` come from the per-dataset blocks in the ORIGINAL TASK section below
and override anything here.

`WANDB_PROJECT` is `ICLR26` for these pixel runs (the earlier latent study used
`THE_GABOR`). Checkpoints go under `THE_GABOR/checkpoints/<run_name>/checkpoints/`
unless a run is launched through `run_alphapre_convlstm.py`, which uses
`Exps/<exp_dir>/<exp_note>/checkpoints/`.

## Deployment

`THE_GABOR/` is **untracked by git** — it will NOT arrive on a server via `git pull`.
Before running anywhere, `rsync` it to that server:

```bash
rsync -az --exclude 'logs/' --exclude '__pycache__/' --exclude 'wandb/' \
  THE_GABOR/ vatsal@<host>:~/NWM/Baselines_Precipitation_Nowcasting/THE_GABOR/
```

Servers: `.88` (2x A6000), `.66` (3x RTX 5000 Ada), `.205` (3x A6000, **GPU0
off-limits**). All three already have the repo, the datasets and the `earthformer`
env. `openpyxl` is required for the results tables (`pip install openpyxl`).

## Evaluation and how the .tex table is updated

`results_table.tex` needs **per-threshold CSI at the two highest intensity
thresholds of each dataset**, which the stock evaluator does not return:

| dataset | thresholds | table columns |
|---|---|---|
| SEVIR | (16, 74, 133, 160, 181, 219) | CSI-181, CSI-219 |
| MeteoNet | (12, 18, 24, 32) | CSI-24, CSI-32 |
| Shanghai | (20, 30, 35, 40) | CSI-35, CSI-40 |
| CIKM | (20, 30, 35, 40) | CSI-35, CSI-40 |

Use **`THE_GABOR/utils/metrics_per_threshold.py::PerThresholdEvaluator`**, a
subclass of `utils.metrics.Evaluator` that re-derives per-threshold CSI from the
counters the base class already accumulates, using its exact formula. It adds
`csi_t<threshold>` for every threshold plus `csi_high` / `csi_high2` (highest and
second-highest). Verified: the averaged metrics are bit-identical to the base
evaluator, and the mean of the per-threshold values reproduces `csi` exactly.
`utils/metrics.py` is not modified.

**Selection rule (decide before running, not after):** a newly trained model
replaces the existing row **only if it improves CSI-M and/or HSS**, and only when
SSIM and PSNR have not degraded badly. If CSI/HSS improve but SSIM/PSNR collapse,
do not silently update — record both in `RUN_STATE.md` and flag it.

Current DAWN-Cast rows to beat (from `baseline_work/results_table.tex`):

| dataset | CSI-M | CSI-H2 | CSI-H | pool4 | pool16 | HSS | SSIM | MSE |
|---|---|---|---|---|---|---|---|---|
| SEVIR | 0.3787 | 0.2177 | 0.1184 | 0.4211 | 0.4862 | 0.4847 | 0.6821 | 340.90 |
| MeteoNet | 0.4529 | 0.4449 | 0.2758 | 0.5159 | 0.6132 | 0.5890 | 0.8419 | 9.31 |
| Shanghai | 0.4525 | 0.4186 | 0.3087 | 0.5113 | 0.5842 | 0.5918 | 0.7301 | 25.33 |
| CIKM | 0.3411 | 0.2424 | 0.1549 | 0.3737 | 0.4282 | 0.4385 | 0.6087 | 34.13 |

## Ablation specification (Part B and Part C)

All variants live in **`THE_GABOR/models/dawncast_ablations.py`**, selected by the
`ablation=` argument of its `get_model()`. Each removes exactly one component;
everything else is imported unchanged from `models/DAWNCast/dawncast.py`.

| item | `ablation` key | what is removed | needs new code? |
|---|---|---|---|
| — | `none` | nothing (baseline) | no |
| a | `a_no_wavelet` | DWT/IDWT; one FAT block on full-resolution features | yes |
| b | `b_shared_fat` | separate HF FAT blocks -> one shared block (`hf_mode='shared'`) | no |
| c | `c_no_gabor` | Gabor stream -> `Linear(T_in, T_out)`; dual-stream + fusion kept so params stay matched | yes |
| d | `d_no_str` | the `str_branch` (`STRModule`) **inside each `SRSTBlock`** | yes |
| e | `e_no_spatial` | the `spatial_branch` (depthwise `Conv2d`) **inside each `SRSTBlock`** | yes |
| f | `f_no_refinement` | the whole SRST stack — **do not run**; take the numbers from Stage-1 training | no |
| f2 | `f_no_srst` | the whole `self.srst` Sequential (2x `SRSTResBlock` + top-level `STRModule`), **Gabor residual bypass retained** | yes |
| g | `g_no_wgtm` | the whole WGTM block -> a single `Linear(T_in, T_out)` over time | yes |

For (d) and (e) it is the branch **inside each block** that is removed; the
top-level `STRModule` following the two `SRSTResBlock`s stays. The other branch,
the GroupNorm, the SiLU and the channel mixing all stay.

**`f_no_srst` vs `f_no_refinement` — both drop SRST, but they are different models.**
`f_no_srst` removes ONLY `self.srst`, so the WGTM output becomes
`reconstructed + gabor_residual`: the Gabor residual bypass is kept, making it a
strict single-component ablation of the refinement block. The Stage-1 model also
omits that residual path, so `f_no_refinement` is "minus SRST **and** minus the
Gabor residual". They have the **same parameter count** (0.422M in the CIKM
config) because the residual bypass adds no parameters — only the computation
differs. Prefer `f_no_srst` for the ablation table; `f_no_refinement` exists only
to reuse Stage-1 numbers at no extra compute.

Verified build (CIKM config, 5->10, db4 J=2, blocks=1, hsf=1, k_spatial=7):
baseline 15.318M params; `a` 15.170M, `b` 15.244M, `c` 15.318M, `d` 8.754M,
`e` 15.193M, `f_no_srst` 0.422M, `f_no_refinement` 0.422M, `g` 0.265M — all
producing `(B, 10, 1, 128, 128)`, and `f_no_srst` confirmed to train
(71 tensors receive non-zero gradients, FACL-only objective intact).

**Ablations do NOT use a Stage-1 donor.** They are ordinary DAWN-Cast runs using
the `freq_multiplier_*` / `weight_scale_*` values given in Part B, with the same
epochs/batch/optimiser as above. Part C uses the **config** of the winning
MeteoNet model (not its weights) with the MeteoNet per-dataset settings.

## Stage-1 donor checkpoints that already exist (Part F)

On **`.88`**, `~/NWM/Baselines_Precipitation_Nowcasting/`:

```
THE_GABOR/checkpoints/Gabor_pixel_SEVIR_storm_seed0/checkpoints/   best step 37260
THE_GABOR/checkpoints/Gabor_pixel_SEVIR_random_seed0/checkpoints/  best step 70000
```

Each contains `best_model.pt`, `gabor_state_best.pt`, `final_model.pt`,
`gabor_state.pt`, `initial_model.pt`, `last_model.pt`; subbands
`['LL', 'HF_level_1', 'HF_level_2']`. Use `gabor_state_best.pt` + `best_model.pt`
as the Part-F donors. rsync them to whichever server runs Part F.

## Seeds (item E)

Any seeds other than 0 — use **1, 2, 3, 4** unless there is a reason not to. Keep
everything else identical to the Part-A recipe so the runs are comparable.

---

# ORIGINAL TASK

So I think you understood about 2 stage training.

NOTE1: Following has to be completed by Sept 7 2026.
NOTE2: From now we will only deal in pixel space.
NOTE3: Use .205 (GPU0 AND GPU2 ONLY FOR ALL THE EXPS, DO NOT USE GPU1 OF .205, ALL GPU'S OF .88 AND .66 CAN BE USED)

A. Experiments (Part A):
Part 1. Training temporal_part + lifting + projection for pixel space dataset.
Part 2. Use the optimal gabor parameters from part1 to initialize gabor in part2 training. (As done earlier)

Then compare the trained model results with preexisting results in
/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/results_table.tex —
if the newly trained model results (after 2 stage training) are better then keep
those results, and update the .tex table, else do not update. And report where
the existing checkpoints are saved.

Running status:
Run CIKM - In .66 server.
Run SEVIR - In .205 (GPU0 and GPU2) multigpu training with total batch size 8 (4 for
each gpu); this violates the batch-size-4 training protocol — that discrepancy will
be specially noted in the paper.
Run Meteonet - In .88 GPU0.
Run Shanghai - In .88 GPU1.

For all CIKM runs keep the following initializations intact:
sparsity_threshold: 0.01
spectral_blocks: 1
spectral_hidden_size_factor: 1
wave: db4
wavelet_level: 2
k_spatial: 7

For all Meteonet runs keep the following initializations intact:
sparsity_threshold: 0.01
spectral_blocks: 4
spectral_hidden_size_factor: 4
wave: db6
wavelet_level: 1
k_spatial: 3

For all Shanghai runs keep the following initializations intact:
sparsity_threshold: 0.01
spectral_blocks: 4
spectral_hidden_size_factor: 3
wave: db6
wavelet_level: 3
k_spatial: 3

For all Sevir runs keep the following initializations intact:
sparsity_threshold: 0.01
spectral_blocks: 4
spectral_hidden_size_factor: 4
wave: db6
wavelet_level: 2
k_spatial: 3

B. Experiments (Part-B, run in parallel with Part-A, in .66 GPUs):
1. Run the ablation studies for the CIKM dataset in .66; for model initialization
   keep:
   freq_multiplier_high: 95.56
   freq_multiplier_low: 22.74
   sparsity_threshold: 0.01
   spectral_blocks: 1
   spectral_hidden_size_factor: 1
   weight_scale_high: 0.25
   weight_scale_low: 0.1
   wave: db4
   wavelet_level: 2

Ablations to run for the required datasets:
a. w/o Wavelet transform — random gabor initialization allowed.
b. Without separate FAT blocks, shared FAT blocks instead — random gabor
   initialization allowed.
c. w/o Gabor — keep parameters matched even after removing gabor; replacement can
   be one MLP with 5 inputs, 10 outputs.
d. w/o STRModule in SRST Block.
e. w/o Spatial Module in SRST Block.
f. w/o full refinement block (no need to run explicitly — results come from Stage 1
   training of CIKM dataset).
g. w/o full WGTM block — instead one MLP for input-to-output projection.

C. Experiments (Part-C, run in sequence with Part-A):
After Meteonet pixel-space training and selecting the best model (comparing the
previous best, stored at
/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/Meteonet/Meteonet_pixel_flow1.09_fhigh1.12/,
against the Part-A trained model), use the best model for the same ablations (a-g
above) on the Meteonet dataset, on GPU0 and GPU1 of the .88 server.

After all CIKM and Meteonet ablations are complete, make a .tex table of ablations
done on the dawncast pixel-space dataset and share the .tex file location.

D. After the Sevir run is complete in .205, run a Meteonet experiment there without
   stage-1 training (random gabor initialization), and similarly for CIKM in .66.

E. If, alongside all the above runs, 4 runs of CIKM and Shanghai on random seeds
   other than seed 0 are possible, do that too. Select the best CIKM and Shanghai
   model (compare previous best vs. new best) — any free GPU can be used, keeping
   the protocol above in mind. Previous Shanghai best model is stored at
   /home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel/Shanghai/Shanghai_pixel_flow1.09_fhigh0.14/.
   Prefer running CIKM in .66, since CIKM can only run there. This item can run
   even after Sept 7.

F. Sevir storm and random pixel-space Stage-1 training already exist. If possible,
   run the following pixel-space experiments (as already done for latent space):

   a. Dawncast on Sevir storm dataset — gabor init from Sevir storm Stage-1
      training in pixel space.

   b. Dawncast on Sevir storm dataset — gabor init from Sevir random Stage-1
      training in pixel space.

   c. Dawncast on Sevir random dataset — gabor init from Sevir storm Stage-1
      training in pixel space.

   d. Dawncast on Sevir random dataset — gabor init from Sevir random Stage-1
      training in pixel space.


NOTE: Report all parameter scores at the end.

## Completion signal
When every item in this prompt (Part A, B, C, and items 4-6) is fully done, all
result tables are updated, and scores are reported — and only then — print the
exact line `ALL_EXPERIMENTS_COMPLETE` on its own line as the last thing you say.
Do not print it prematurely; an automated script is watching for it to know when
to stop restarting you.