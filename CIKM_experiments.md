# CIKM — Gabor-Initialized Parameter Minimization

**Self-contained runbook.** Everything needed to execute this is below; no prior conversation context is required.

**Goal:** cut DAWN-Cast's trainable parameter count on CIKM (pixel space) as far as possible **without losing test performance** against the 15,318,149-parameter Gabor-sweep baseline. Every experiment's parameter count and metrics get appended to `Parameter_budget.csv`.

- **Repo root:** `/home/vatsal/NWM/Baselines_Precipitation_Nowcasting`
- **Runner:** `run_alphapre_convlstm.py` (do **not** modify)
- **Model:** `models/DAWNCast/dawncast.py` (do **not** modify)
- **Run everything from the repo root.**

---

## 1. Gabor initialization values (CIKM)

**Source run:** `Exps/Gabor_sweep_runs/CIKM_pixel_flow22.74_fhigh95.56`
Read from that run's `params.yaml`. It is the only CIKM run in `Exps/Gabor_sweep_runs/`, i.e. the sweep's selected best; its `ckpt-best.pt` was chosen on validation CSI (best 0.35006).

| Parameter | Value |
|---|---|
| `weight_scale_low` | `0.1` |
| `alpha_low` | `1.0` |
| `beta_low` | `43.1034` |
| `freq_multiplier_low` | `22.74` |
| `weight_scale_high` | `0.25` |
| `alpha_high` | `1.0` |
| `beta_high` | `4.8193` |
| `freq_multiplier_high` | `95.56` |

**These eight values are frozen across every experiment below.** They are the initialization, not something to tune here.

---

## 2. Baseline (CIKM-B0) — establish and log this BEFORE running anything

The baseline run already exists and does **not** need retraining. Its numbers:

**Architecture:** `num_blocks(--spectral_blocks)=1`, `hidden_size_factor(--spectral_hidden_size_factor)=1`, `wavelet_level=2`, `wave=db4`, `conv_kernel(k_spatial)=7`, `hidden_dim=64`, `hf_mode=separate`, `size_factor=1.0`, `T_in=5`, `T_out=10`, `img_size=128`, `img_channel=1`, `seq_len=15`.

**Parameter count: 15,318,149 (15.32M)** — verified by rebuilding the model and counting `sum(p.numel() for p in model.parameters() if p.requires_grad)`; matches the `Main Model Parameters: 15.32M` line in its log.

**Test metrics (from `ckpt-best.pt`):**

| metric | value |
|---|---|
| **csi** | **0.34046** |
| csi4 (pooled 4×4) | 0.37582 |
| csi16 (pooled 16×16) | 0.43476 |
| hss | 0.43828 |
| mse | 34.45998 |
| mae | 3.20824 |
| rmse | 5.54528 |
| psnr | 23.74819 |
| ssim | 0.60329 |
| crps | 0.04010 |
| lpips | 0.35822 |

**Best validation CSI: 0.35006 at epoch 10** (validation runs every 5 epochs).

### Seed the baseline row first

```bash
cd /home/vatsal/NWM/Baselines_Precipitation_Nowcasting
conda run -n earthformer python3 scripts/param_budget_log.py seed-baseline \
    --run_dir Exps/Gabor_sweep_runs/CIKM_pixel_flow22.74_fhigh95.56 \
    --dataset cikm --config_id CIKM-B0 \
    --gabor_source "Exps/Gabor_sweep_runs/CIKM_pixel_flow22.74_fhigh95.56" \
    --notes "Gabor sweep best run; parameter-budget reference."
```

(If `Parameter_budget.csv` already contains a `cikm/CIKM-B0` row, this updates it in place rather than duplicating.)

---

## 3. What "no compromise" means numerically

Primary metric: **test CSI** (mean over thresholds), computed from `ckpt-best.pt`.

- **Empirical eval noise:** two evaluations of this baseline logged CSI `0.34108` and `0.34046`, a spread of **0.00062 (0.18% relative)**. Treat anything inside ±0.0006 as indistinguishable.
- **PASS (no compromise):** `test_csi >= 0.33706` — within **1% relative** of the 0.34046 baseline.
- **STRONG PASS:** `test_csi >= 0.34046` — meets or beats baseline outright.
- **FAIL:** `test_csi < 0.33706`.

**Secondary guards** — a config that passes on CSI but regresses badly elsewhere is not a pass. Require **both**:
- `test_csi16 >= 0.43041` (within 1% of 0.43476)
- `test_hss >= 0.43390` (within 1% of 0.43828)

**Objective:** among all configs that PASS, take the one with the **smallest `param_count`**.

> **Expect reductions to help, not just save memory.** The baseline peaks at validation epoch 10 of 50 and then degrades — it is over-parameterized for CIKM and overfits. Smaller configs beating the baseline is a plausible outcome, not a surprise.

---

## 4. Environment split

| Phase | Conda env | Flag |
|---|---|---|
| Train + validate | `earthformer` | `--valid` |
| Test | `alphapre_manual` | `--eval` |

Both envs are verified working (torch 2.7.0+cu118, CUDA available, `lpips` and `utils.metrics` import cleanly). Invoke as `conda run --no-capture-output -n <env> python3 ...`.

---

## 5. Experiment list and order

One axis matters. Parameter count on CIKM is **exactly**:

```
params = 6,943,129
       + 8,192,000 * (hidden_size_factor / num_blocks)
       +     6,400 * hidden_size_factor
       +    74,220 * (wavelet_level - 1)
       +   102,400   if conv_kernel == 7 (vs 3)
```

(Verified against instantiated models, zero mismatches. `hidden_size = hidden_dim * T_out = 64*10 = 640`, and **`num_blocks` must divide 640** — it is a hard `assert` in `STRModule`. Valid divisors: 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 320, 640.)

So **`hidden_size_factor / num_blocks` is the capacity knob** — it carries 8.19M of the baseline's 15.32M. The floor with that term driven to zero is ~7.05M.

### Phase 1 — capacity ladder (run these five, in this order)

Everything except `--spectral_blocks` is held at baseline. Order brackets the interesting region first.

| # | config_id | `--spectral_blocks` | `--spectral_hidden_size_factor` | ratio | params | % of baseline |
|---|---|---|---|---|---|---|
| 1 | `CIKM-R2` | 4 | 1 | 1/4 | 9,174,149 | 59.9% |
| 2 | `CIKM-R4` | 16 | 1 | 1/16 | 7,638,149 | 49.9% |
| 3 | `CIKM-R3` | 8 | 1 | 1/8 | 8,150,149 | 53.2% |
| 4 | `CIKM-R1` | 2 | 1 | 1/2 | 11,222,149 | 73.3% |
| 5 | `CIKM-R5` | 64 | 1 | 1/64 | 7,254,149 | 47.4% |

**Stop rule:** if `CIKM-R2` and `CIKM-R4` both PASS, `CIKM-R1` (#4) is redundant — skip it and go straight to `CIKM-R5`. If `CIKM-R4` FAILS but `CIKM-R2` PASSES, run `CIKM-R3` to bisect and skip `CIKM-R5`.

### Phase 2 — cheap extras (only on the smallest Phase-1 config that PASSED)

Apply to the winner; each is an independent one-run test.

| config_id | change from winner | saving | params if applied to CIKM-R2 |
|---|---|---|---|
| `CIKM-K3` | `--conv_kernel 3` (from 7) | −102,400 | 9,071,749 |
| `CIKM-L1` | `--wavelet_level 1` (from 2) | −74,220 | 9,099,929 |

These are small savings. Run them only if Phase 1 leaves budget; they mainly test whether the baseline's `k=7` and `level=2` are earning their keep.

**Total: 5 Phase-1 runs + up to 2 Phase-2.** At ~26.8 min/epoch × 50 epochs, budget **~22.3 hours per run** on one GPU.

---

## 6. Exact commands

Set these once per shell:

```bash
cd /home/vatsal/NWM/Baselines_Precipitation_Nowcasting

GPU=0                       # change to the GPU you were allocated
EXP_DIR=cikm_param_reduction

# Gabor init — frozen, CIKM sweep-best
WS_LOW=0.1   ; WS_HIGH=0.25
A_LOW=1.0    ; A_HIGH=1.0
B_LOW=43.1034; B_HIGH=4.8193
F_LOW=22.74  ; F_HIGH=95.56
```

### Template — run one config

Substitute `CONFIG_ID`, `NB`, `HSF`, and (Phase 2 only) `LEVEL` / `KERNEL`.

```bash
CONFIG_ID=CIKM-R2 ; NB=4 ; HSF=1 ; LEVEL=2 ; KERNEL=7
NOTE=${CONFIG_ID}_nb${NB}_hsf${HSF}

# ---------- TRAIN + VALIDATE  (env: earthformer) ----------
CUDA_VISIBLE_DEVICES=${GPU} conda run --no-capture-output -n earthformer \
python3 run_alphapre_convlstm.py \
    --backbone                    DAWNCast \
    --seed                        0 \
    --exp_dir                     ${EXP_DIR} \
    --exp_note                    ${NOTE} \
    --dataset                     cikm \
    --img_size                    128 \
    --img_channel                 1 \
    --seq_len                     15 \
    --frames_in                   5 \
    --frames_out                  10 \
    --num_workers                 8 \
    --wave                        db4 \
    --wavelet_level               ${LEVEL} \
    --hf_mode                     separate \
    --weight_scale_low            ${WS_LOW} \
    --alpha_low                   ${A_LOW} \
    --beta_low                    ${B_LOW} \
    --freq_multiplier_low         ${F_LOW} \
    --weight_scale_high           ${WS_HIGH} \
    --alpha_high                  ${A_HIGH} \
    --beta_high                   ${B_HIGH} \
    --freq_multiplier_high        ${F_HIGH} \
    --spectral_blocks             ${NB} \
    --spectral_hidden_size_factor ${HSF} \
    --sparsity_threshold          0.01 \
    --conv_kernel                 ${KERNEL} \
    --hidden_dim                  64 \
    --size_factor                 1.0 \
    --epochs                      50 \
    --batch_size                  4 \
    --lr                          1e-4 \
    --wandb_state                 online \
    --wandb_project_name          DAWNCast_param_budget \
    --run_name                    cikm_${NOTE} \
    --gpu_use                     ${GPU} \
    --valid

# ---------- TEST  (env: alphapre_manual) ----------
CUDA_VISIBLE_DEVICES=${GPU} conda run --no-capture-output -n alphapre_manual \
python3 run_alphapre_convlstm.py \
    --backbone                    DAWNCast \
    --seed                        0 \
    --exp_dir                     ${EXP_DIR} \
    --exp_note                    ${NOTE} \
    --dataset                     cikm \
    --img_size                    128 \
    --img_channel                 1 \
    --seq_len                     15 \
    --frames_in                   5 \
    --frames_out                  10 \
    --num_workers                 8 \
    --wave                        db4 \
    --wavelet_level               ${LEVEL} \
    --hf_mode                     separate \
    --weight_scale_low            ${WS_LOW} \
    --alpha_low                   ${A_LOW} \
    --beta_low                    ${B_LOW} \
    --freq_multiplier_low         ${F_LOW} \
    --weight_scale_high           ${WS_HIGH} \
    --alpha_high                  ${A_HIGH} \
    --beta_high                   ${B_HIGH} \
    --freq_multiplier_high        ${F_HIGH} \
    --spectral_blocks             ${NB} \
    --spectral_hidden_size_factor ${HSF} \
    --sparsity_threshold          0.01 \
    --conv_kernel                 ${KERNEL} \
    --hidden_dim                  64 \
    --size_factor                 1.0 \
    --batch_size                  4 \
    --wandb_state                 online \
    --wandb_project_name          DAWNCast_param_budget \
    --run_name                    cikm_${NOTE} \
    --gpu_use                     ${GPU} \
    --eval \
    --ckpt_milestone              Exps/${EXP_DIR}/${NOTE}/checkpoints/ckpt-best.pt

# ---------- RECORD ----------
conda run -n earthformer python3 scripts/param_budget_log.py append \
    --run_dir        Exps/${EXP_DIR}/${NOTE} \
    --dataset        cikm \
    --config_id      ${CONFIG_ID} \
    --baseline_params 15318149 \
    --gabor_source   "Exps/Gabor_sweep_runs/CIKM_pixel_flow22.74_fhigh95.56" \
    --status         complete \
    --notes          "nb=${NB} hsf=${HSF} lvl=${LEVEL} k=${KERNEL}; reduction vs CIKM-B0"
```

### Per-config substitutions

```bash
# Phase 1
CONFIG_ID=CIKM-R2 ; NB=4  ; HSF=1 ; LEVEL=2 ; KERNEL=7
CONFIG_ID=CIKM-R4 ; NB=16 ; HSF=1 ; LEVEL=2 ; KERNEL=7
CONFIG_ID=CIKM-R3 ; NB=8  ; HSF=1 ; LEVEL=2 ; KERNEL=7
CONFIG_ID=CIKM-R1 ; NB=2  ; HSF=1 ; LEVEL=2 ; KERNEL=7
CONFIG_ID=CIKM-R5 ; NB=64 ; HSF=1 ; LEVEL=2 ; KERNEL=7

# Phase 2 (apply to the smallest Phase-1 PASS; NB/HSF shown for CIKM-R2)
CONFIG_ID=CIKM-K3 ; NB=4  ; HSF=1 ; LEVEL=2 ; KERNEL=3
CONFIG_ID=CIKM-L1 ; NB=4  ; HSF=1 ; LEVEL=1 ; KERNEL=7
```

### Verify a parameter count before committing 22 hours

```bash
conda run -n earthformer python3 -c "
from models.DAWNCast.dawncast import get_model
m = get_model(afno_blocks=4, sparsity_threshold=0.01, afno_hidden_size_factor=1,
    weight_scale_low=0.1, alpha_low=1.0, beta_low=43.1034, freq_multiplier_low=22.74,
    weight_scale_high=0.25, alpha_high=1.0, beta_high=4.8193, freq_multiplier_high=95.56,
    size_factor=1.0, k_spatial=7, img_channels=1, dim=64, T_in=5, T_out=10,
    wave='db4', wavelet_level=2, hf_mode='separate', input_shape=(128,128))
print(f'{sum(p.numel() for p in m.parameters() if p.requires_grad):,}')"
# -> 9,174,149
```

---

## 7. Output locations

For `--exp_dir cikm_param_reduction --exp_note ${NOTE}`, the runner creates:

```
Exps/cikm_param_reduction/${NOTE}/
├── checkpoints/
│   ├── ckpt-best.pt      <- best validation CSI; USE THIS FOR TEST
│   └── ckpt-last.pt
├── logs/log.log          <- "Main Model Parameters", "Valid Results", "Test Results"
├── valid_samples/
├── test_samples/
├── sanity_check/
└── params.yaml           <- full arg dump (rewritten on every invocation)
```

Also written:
- **`Parameter_budget.csv`** (repo root) — the deliverable, via `scripts/param_budget_log.py`.
- `/home/vatsal/Dataserver2/Neurips/csv_files/Rebuttal_runs.csv` — the runner's own shared results CSV, appended automatically on `--eval`. Harmless side effect; not the deliverable.
- WandB project `DAWNCast_param_budget`.

---

## 8. How results map into `Parameter_budget.csv`

`scripts/param_budget_log.py append` does this automatically — it reads `params.yaml` and `logs/log.log` from the run dir, **rebuilds the model to recount parameters exactly** (it does not trust the rounded `15.32M` log line), and upserts one row keyed on `(dataset, config_id)`. Re-running a config updates its row instead of duplicating it.

Columns: `dataset`, `config_id`, `status`, `param_count`, `param_pct_of_baseline`, `backbone`, `num_blocks`, `hidden_size_factor`, `level`, `wave`, `k_spatial`, `hidden_dim`, `hf_mode`, `T_in`, `T_out`, `epochs`, the eight Gabor values, `gabor_source_run`, `best_val_csi`, `best_val_epoch`, `test_csi`, `test_csi4`, `test_csi16`, `test_hss`, `test_mse`, `test_mae`, `test_rmse`, `test_psnr`, `test_ssim`, `test_crps`, `test_lpips`, `checkpoint_path`, `log_path`, `notes`.

Inspect progress at any time:

```bash
column -s, -t < Parameter_budget.csv | cut -c1-160
```

---

## 9. Gotchas — read before running

1. **`--frames_out 10` must be passed explicitly.** `Runner.__init__` forces `frames_in=5 / frames_out=10` for `cikm`, but only *after* `_build_model()` has already run. Without the flag you build a `T_out=20` model and feed it `T_out=10` batches. This also makes `hidden_size = 640`, which every parameter number here depends on.

2. **`num_blocks` must divide 640** — hard `assert` in `STRModule`. Sticking to the table above is safe.

3. **Use `--backbone DAWNCast`, not `DAWNCast_old`.** The baseline was trained with `DAWNCast_old` (`dawncast_old.py`), but the two files are **numerically identical** — same parameter count (15,318,149) and same tensor shapes; the only difference is the module attribute rename `lastocast.*` → `dawncast.*`. Training from scratch is unaffected. **But the baseline checkpoint will not load into `DAWNCast`** because of the key prefix, so do not try to resume or finetune from `CIKM-B0`'s `ckpt-best.pt` without remapping keys.

4. **Always test from `ckpt-best.pt`, explicitly.** Pass `--ckpt_milestone .../ckpt-best.pt`. Without it, `check_milestones()` tries to parse numeric milestones out of the checkpoint filenames, which does not match the `ckpt-best` / `ckpt-last` naming produced under `--valid`.

5. **Validation runs only every 5 epochs** (`if (epoch+1)%5==0`), and `ckpt-best.pt` updates only at those points.

6. **`params.yaml` is overwritten on every invocation** of the same `exp_note`, including the `--eval` pass. After eval it will show `eval: true`. That is expected; the Gabor values in it stay correct.

7. **`--epochs 50`** keeps the cosine LR schedule (warmup = 20% of total steps) identical to the baseline, so `ckpt-best` selection is apples-to-apples. Changing the epoch count changes the LR schedule and breaks the comparison.

---

## 10. Reporting

When Phase 1 (and any Phase 2) is done:

1. Confirm `Parameter_budget.csv` has a row per config with a non-empty `test_csi`.
2. Apply the §3 rule: among PASS configs, report the one with the smallest `param_count`, as **"CIKM: X params (Y% of baseline), test CSI Z vs baseline 0.34046"**.
3. Flag any config that **beat** the baseline — given the epoch-10 overfitting, that is a likely and reportable result.
