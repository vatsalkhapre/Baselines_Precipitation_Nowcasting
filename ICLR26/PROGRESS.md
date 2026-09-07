# ICLR26 — DAWN-Cast pixel-space campaign: what has been done

Working record of the campaign defined in
`PROMPT/dawncast_pixel_prompt.md`. Home for all further activity is
`~/NWM/Baselines_Precipitation_Nowcasting/ICLR26/` on **.66**.

Last updated: 2026-09-07.

---

## 0. Scope

Everything is **pixel space** (128x128, `T_in=5`). Four datasets: SEVIR,
MeteoNet, Shanghai, CIKM. Objective is **FACL only** in both stages
(`utils.utilspp.RandomScheduling`, imported unmodified). 50 epochs, seed 0
unless stated, `earthformer` conda env on every server.

Servers: `.88` (2x A6000) · `.66` (3x RTX 5000 Ada) · `.205` (3x A6000,
**GPU1 must not be used**).

---

## 1. What "two-stage training" means here

**Stage 1** — a reduced model containing only the lifting, the wavelet temporal
path and the projection:

```
Input -> Lifting -> DWT -> per subband (Gabor + MLP -> fusion) -> IDWT -> Projection
```

No SRST / STR / AFNO, no Gabor residual bypass. Gabor initialised neutrally:
`freq_multiplier = 1.0` on every subband, `freq ~ U(0,1)`, one shared
`weight_scale/alpha/beta`. Model `THE_GABOR/models/gabor_mlp_model.py`,
runner `THE_GABOR/run_stage1_pixel.py`.

**Stage 2** — the full DAWN-Cast, with each subband's Gabor initialised from the
Stage-1 **best-validation** checkpoint of the same dataset.
Model `THE_GABOR/models/dawncast_transfer.py`, runner
`THE_GABOR/run_stage2_pixel.py`.

Two constraints that silently corrupt the transfer if broken:
1. `hf_mode` must be `separate` (a donor has `1+J` subbands; `shared` has 2).
2. The donor's `freq_multiplier` (1.0) must be carried over. It is a plain float,
   not a learnable tensor, and the Gabor computes
   `sin(freq_multiplier * freq * linear(x))`; DAWN-Cast's own default is 4.0, so
   leaving it would rescale every transferred frequency 4x.

---

## 2. Workspace

```
ICLR26/
  PROGRESS.md          this file
  make_tables.py       builds both .tex tables from the JSON in eval/
  eval/                one <run_name>.json of test metrics per run
  tables/
      twostage_vs_best.tex   4 datasets: 2-stage vs published DAWN-Cast + % change
      ablations.tex          CIKM + MeteoNet ablations + % change
  logs/                evaluation and finalisation logs
```

Training code lives in `THE_GABOR/` (synced to all three servers; it is **not**
tracked by git, so it must be rsynced, never `git pull`ed).

---

## 3. Code written for this campaign

| file | purpose |
|---|---|
| `THE_GABOR/models/gabor_mlp_model.py` | Stage-1 controlled model |
| `THE_GABOR/models/dawncast_transfer.py` | full DAWN-Cast with **per-subband** Gabor params (the original interpolates HF frequencies, so subbands cannot be addressed individually). Verified identical to the original: same 127-tensor `state_dict`, 0.0 output difference |
| `THE_GABOR/models/dawncast_ablations.py` | ablation variants a-g plus `f_no_srst` |
| `THE_GABOR/datasets/pixel_datasets.py` | generic pixel loaders for all 4 datasets (+ SEVIR regime filtering for Part F) |
| `THE_GABOR/datasets/sevir_regime_dataset.py` | SEVIR RANDOM/STORM catalog mask |
| `THE_GABOR/utils/gabor_transfer.py` | donor -> DAWN-Cast transfer + freezing, shape-checked, fails loudly |
| `THE_GABOR/utils/metrics_per_threshold.py` | per-threshold CSI (`csi_t*`) on top of `utils/metrics.py`, which only returns the threshold average. Needed for the CSI-181/219-style columns |
| `THE_GABOR/eval_pixel.py`, `eval_many.py` | test evaluation, one JSON per run |
| `THE_GABOR/select_best_init.py` | Part C gate: previous-best vs new MeteoNet |
| `THE_GABOR/select_best_seed.py` | item E: rank seeds, compare to published best |
| `THE_GABOR/orchestrate.sh` | self-driving per-GPU job chains |
| `THE_GABOR/health.sh` | 30-min health snapshot |
| `ICLR26/make_tables.py` | the two LaTeX tables |

`models/DAWNCast/dawncast.py` and every other pre-existing file are **unmodified**
(verified by sha256 against `THE_GABOR/configs/repo_baseline.txt`).

---

## 4. Experiment inventory

### Part A — two-stage training (COMPLETE)

| dataset | Stage 1 | Stage 2 | server |
|---|---|---|---|
| CIKM | done (50/50) | done (50/50) | .66 |
| MeteoNet | done | done | .88 GPU0 |
| Shanghai | done | done | .88 GPU1 |
| SEVIR | done | done | .205 GPU0+GPU2, DataParallel, total batch 8 (4/GPU) |

Per-dataset architecture (as specified in the task):

| dataset | wave | level | spectral_blocks | hidden_size_factor | k_spatial |
|---|---|---|---|---|---|
| CIKM | db4 | 2 | 1 | 1 | 7 |
| MeteoNet | db6 | 1 | 4 | 4 | 3 |
| Shanghai | db6 | 3 | 4 | 3 | 3 |
| SEVIR | db6 | 2 | 4 | 4 | 3 |

Horizons: CIKM `5 -> 10`; the other three `5 -> 20`.

### Part B — CIKM ablations (COMPLETE, 7/7, on .66)

`a_no_wavelet`, `b_shared_fat`, `c_no_gabor`, `d_no_str`, `e_no_spatial`,
`f_no_srst`, `g_no_wgtm` — all 50/50. Init as specified:
`freq_multiplier` `[22.74, 95.56, 59.15]`, `weight_scale` `[0.1, 0.25, 0.175]`
(the low/high pair expanded with DAWN-Cast's own per-level interpolation).
`f_no_refinement` is free — it is the Stage-1 model.

### Part C — MeteoNet ablations (6/7, on .88)

Gated on a previous-vs-new comparison, which ran and chose the **previous best**,
so the ablations use that model's init
(`flow 1.09 / fhigh 1.12`, `ws 0.1/1.0`, `beta 0.0995/0.1643`).
Done: a, b, c, d, f_no_srst, g. Running: `e_no_spatial` (45/50).

### Part D — no-Stage-1 controls

CIKM done (.66). MeteoNet running on .205 (14/50).

### Part E — extra seeds

CIKM seeds 1-4 done; selection run. Shanghai seeds 1-4 queued on .88 GPU1.

### Part F — SEVIR storm/random Stage-2 (x4)

Queued on .205 behind item D. Donors
`Gabor_pixel_SEVIR_{storm,random}_seed0` already exist and are synced.

---

## 5. Results so far

Reference = the published DAWN-Cast rows of `baseline_work_ICLR26/results_table.tex`
(verified identical to the `baseline_work` copy).

| dataset | reference CSI-M / HSS | two-stage | verdict |
|---|---|---|---|
| MeteoNet | 0.4529 / 0.5890 | **0.4494 / 0.5864** | no improvement -> keep previous |
| CIKM (best of 5 seeds) | 0.3411 / 0.4385 | **0.3313** (seed 3) | no improvement -> keep previous |

CIKM seed sweep: 0.3313 (s3), 0.3237 (s4), 0.3202 (s0), 0.3176 (s2), 0.3150 (s1).
SEVIR and Shanghai evaluations are in progress.

**So far the two-stage model does not beat the published best on either dataset
evaluated**, and `results_table.tex` has therefore not been modified.

Selection rule used: improve CSI-M **or** HSS, vetoed if SSIM drops more than 10%.

---

## 6. Bugs found and fixed (all mine unless noted)

1. **`run_pixel.py` was SEVIR-only** — it routed through the RANDOM/STORM catalog
   mask, so CIKM/MeteoNet/Shanghai could not run. Wrote `pixel_datasets.py` +
   `run_stage1_pixel.py`.
2. **Ablations changed the subband count** (`a_no_wavelet` has 1, `b_shared_fat`
   has 2) but received the 3-entry baseline list -> `_fit_per_subband()`.
3. **The init-checkpoint signature ignored `ablation`**, so all seven ablations
   resolved to one filename and failed a strict load -> signature now includes the
   model/ablation name.
4. **`c_no_gabor` crashed** — it swaps the Gabor for `nn.Linear`, which has no
   `freq_multiplier`, but was still handed to the Gabor logger ->
   `gabor_layers()` returns empty when there is no Gabor.
5. **`.205` CUDA OOM** — DAWN-Cast at 128x128 with DataParallel peaked ~30.6 GiB
   while another user held 16.7 GiB on GPU0. Now gated on both GPU0 and GPU2
   being free (`wait_gpus_free`), plus `expandable_segments`.
6. **The seed-selection step raced sibling chains** — it ran while other chains
   still held the GPUs, so every evaluation died with OOM / "device busy". Now
   gated on a free GPU.
7. **THE IMPORTANT ONE — evaluation rebuilt the wrong network.**
   `wandb_config()` never persisted `afno_blocks`, `afno_hidden_size_factor`,
   `sparsity_threshold` or `k_spatial`, so `eval_pixel` fell back to generic
   defaults (4/4/3) and constructed a *different* model than was trained:
   `srst...str_branch.w1` checkpoint `[2,1,640,640]` vs rebuilt `[2,4,160,640]`;
   `spatial_branch` `[640,1,7,7]` vs `[640,1,3,3]`.
   CIKM (1/1/7) and Shanghai (4/3/3) could never load; MeteoNet and SEVIR only
   worked because their values happen to equal the defaults — meaning the Part C
   MeteoNet gate was right by luck, not by construction.
   Fixed both ways: the four values are now persisted, and `eval_pixel` recovers
   them from the dataset for checkpoints already written, raising rather than
   silently defaulting.
8. **`models/DAWNCast/dawncast.py` was corrupted by a stray IDE keystroke**
   (`x.shape` -> `x.shapeIn`, a syntax error). Not part of this work; restored to
   the committed content, broken copy kept under
   `THE_GABOR/logs/_recovered/`.

---

## 7. Infrastructure

* **`orchestrate.sh`** — one detached chain per GPU. Idempotent (`run_job` skips
  any run whose log already contains `[done]`), chains behind an already-running
  job, archives failed logs to `logs/_runlogs/_failed/`, and never aborts the
  whole chain on a single failure.
* **cron on all three servers** — `@reboot` relaunch, a `*/30` re-arm (restarts a
  chain that died *or* exited with failures inside), and a `*/30` health snapshot
  to `THE_GABOR/logs/_runlogs/HEALTH.txt`.
  Worst-case idle after a silent stop is ~30 min.
* Two real outages were absorbed: `.88` rebooted on 09-01 (cost ~24 GPU-hours
  *before* cron was installed) and `.66` was rebooted on 09-06 after its GPU
  driver wedged — the `@reboot` cron re-armed the chains and every finished run
  was correctly skipped.

---

## 8. Open issues

1. **The ablation table has no matched full-model control.** Variants a-g were
   trained with the Part-B init, but no `--ablation none` run exists with that
   same init, so the percentages are currently taken against the published
   DAWN-Cast row, which used a different initialisation. One extra run per dataset
   (~21 h for CIKM) would make them a clean single-variable comparison.
2. Shanghai has not been through the seed sweep or its selection yet.
3. Part F has not started (queued behind item D on .205).
4. Nothing alerts on repeated failure; `HEALTH.txt` and `_failed/` record it, but
   someone has to look.

---

## 9. Resuming

1. Read `baseline_work/RUN_STATE.md` (chronological log) and this file.
2. `bash THE_GABOR/orchestrate.sh <chain>` is safe to re-run at any time; it skips
   finished work.
3. `python -m THE_GABOR.eval_many --runs ... --out_dir ICLR26/eval` to evaluate.
4. `python ICLR26/make_tables.py` to rebuild both tables from whatever JSON exists;
   missing runs render as "evaluation pending" rather than failing.
