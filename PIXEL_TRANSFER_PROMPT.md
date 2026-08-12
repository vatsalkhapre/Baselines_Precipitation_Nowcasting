# Prompt for a fresh Claude Code session (pixel-space transfer experiments)

Copy everything below the line into the new session. It assumes no prior context.

---

I need you to run a set of pixel-space transfer-learning experiments for the DAWN-Cast
precipitation nowcasting model. Everything is already written and verified — your job is to
**run it, watch it, and report the results**, not to redesign anything. Be conservative: verify
before you launch, and do not modify any production file.

## Working directory

`/home/vatsal/NWM/Baselines_Precipitation_Nowcasting`

## What the experiment is

A SEVIR-pretrained DAWN-Cast model is transferred to two target datasets (MeteoNet, Shanghai) by
**freezing the whole network except one small "adaptation surface"** and fine-tuning only that.
Everything is pixel space: 128×128, single channel, 5 input frames → 20 output frames. No
autoencoder is involved anywhere (that is the latent-space variant, not this one).

Pretrained source checkpoint (must exist, ~950 MB):
`/home/vatsal/Dataserver2/Neurips/DAWNCast_pixelspace/dawncast_sevir_pixel/checkpoints/ckpt-best.pt`

## Files (all already written — read them before running)

| File | Role |
|---|---|
| `finetune_temporal_path_transfer_pixel.py` | The runner. Subclasses `Runner` from `run_alphapre_convlstm.py`, loads the pretrained checkpoint, freezes everything, re-opens the chosen surface, audits, trains, saves an adapter checkpoint. |
| `scripts/scripts_run/run_dawncast_transfer_sweep_pixel.sh` | One run = one (dataset, config, gpu, lr, train_frac). Trains 50 epochs then auto-evaluates the best checkpoint. |
| `scripts/scripts_run/run_all_pixel_transfer_experiments.sh` | **The orchestrator — this is what you run.** Runs all experiments in order, picks the best surface from the CSV, then runs the data-fraction experiments on it, and prints a final table. |
| `finetune_temporal_path_transfer.py` | Latent-space sibling. The pixel runner imports its audit/freeze helpers. **Do not edit it.** |

## The experiments

Each on **both** MeteoNet and Shanghai:

| Config | Surface unfrozen | Trainable |
|---|---|---|
| `zeroshot` | nothing — pretrained evaluated as-is (the floor) | 0 |
| `liftproj` | temporal + lifting + projection | 423,009 |
| `normbias` | temporal + norms + biases, **whole model** | 249,761 |
| `normbias_stem` | temporal + norms + biases, **`srst` excluded** | 160,161 |
| `liftprojonly` | lifting + projection only, temporal frozen | 264,513 |

Then the best of those four (by mean test CSI across both datasets) is re-run on **50%** and
**20%** of the training set. Model total is 59,541,089 parameters, so every surface is under 1%.

"temporal" = the `gabor` + `mlp` + `fusion` of every `FATBlock` inside the `WGTMBlock`, i.e. all
temporal processing above the IDWT reconstruction. `srst` is the large frozen spectral trunk.

## How to run it

```bash
cd /home/vatsal/NWM/Baselines_Precipitation_Nowcasting
bash scripts/scripts_run/run_all_pixel_transfer_experiments.sh 0 1 2   # three GPU indices
```

Pass whichever GPU indices are free (`nvidia-smi` first — each run needs ~14 GB). Passing the same
index three times serialises everything onto one GPU. It runs ~12–16 h on three GPUs, so start it
in the background and poll, rather than blocking on it:

```bash
setsid nohup bash scripts/scripts_run/run_all_pixel_transfer_experiments.sh 0 1 2 \
    > /tmp/pixel_transfer_orchestrator.log 2>&1 &
```

Per-run logs land in `Exps/transfer_sweep_pixel/_logs/`.

## Before you launch — verify these five things

1. The pretrained checkpoint exists and is ~950 MB.
2. `bash -n` passes on both shell scripts and `python -c "import ast; ast.parse(open(...).read())"`
   passes on the runner.
3. GPUs have ≥14 GB free each.
4. Nothing else of this experiment is already running:
   `ps -eo cmd | grep -c "[f]inetune_temporal_path_transfer_pixel.py"` should be 0.
5. Do a single cheap smoke test first — the zero-shot run, which trains nothing and takes ~15 min:
   `bash scripts/scripts_run/run_dawncast_transfer_sweep_pixel.sh meteo zeroshot 0`
   It must print a `FREEZE AUDIT` block, then `Test Results: {...}`, then
   `[ResultsLogger] Results logged to .../Transfer_runs_pixel.csv`. If that works end to end, the
   whole pipeline works.

## What "healthy" looks like in the logs

* `Loaded pretrained weights from ... (epoch 18, step 113107)` — the checkpoint loaded `strict=True`.
* A `FREEZE AUDIT` block whose `trainable params` matches the table above **exactly**. If it does
  not, stop and report — do not let it train.
* `Optimizer over N tensors / <trainable> params`.
* `Freeze check passed at step 10: <N> frozen tensors bit-identical` — this is the guard that
  proves nothing outside the chosen surface is moving. **If this raises, stop everything and
  report it.** It means the freezing is broken and every number would be meaningless.
* `Sample shape torch.Size([4, 25, 1, 128, 128])` — pixel shapes, not `[4, 25, 4, 32, 32]`
  (that would mean it picked up the latent dataset by mistake).
* Validation runs every 5 epochs; checkpoints and adapters are only written then.

## Known failure modes, and what they mean

* **`FROZEN PARAMETERS MOVED`** — a real bug, not a warning. Stop, report, do not continue.
* **`WARNING: Gabor gamma went non-positive`** — `GaborLayer.gamma` is a trainable unconstrained
  parameter entering as `exp(-0.5·D·γ)`; if it goes negative the envelope expands instead of
  decaying and the run diverges. Report it; the run's numbers are suspect from that point on.
* **`FileNotFoundError: .../ckpt-best.pt`** in the eval phase — the training phase died before
  epoch 5, so no best checkpoint was ever written. Look further up the log for the real error.
* **`Checkpoint is missing N keys the model expects`** — wrong `--backbone`. The pixel checkpoint
  needs `DAWNCast` (the `dawncast.py` naming: `dawncast.wgtm.fat_ll`, `srst`, `spatial_branch`),
  **not** `DAWNCast_old` (the `dawncast_old.py` naming: `lastocast.operator.stream_ll`,
  `conv_spectral`, `dw_spatial`). The orchestrator already passes the right one.
* **CIKM** cannot be a target: its sequences are 15 frames (5→10) and the checkpoint is 5→20, so
  the shape check rejects it on purpose. Only MeteoNet and Shanghai work.

## Results

Everything lands in `/home/vatsal/Dataserver2/Neurips/csv_files/Transfer_runs_pixel.csv`, one
**final test row per run** (no per-validation rows). In that CSV:

* `Experiment Details` = `pixeltransfer_<config>_<dataset>_lr1e-4[_fracNN]`
* `Model Params (in M)` column holds the **trainable** parameter count for that run, not the model
  total (the transfer budget is the number that matters here).
* `Why?` holds `trainable N/59,541,089 (x.xxxx%) | lr ... | train_frac ...` so the two can never be
  confused.
* Metrics: CSI-M, CSI-4, CSI-16, HSS, SSIM, MSE, PSNR, MAE, RMSE, CRPS, LPIPS.

The orchestrator prints a formatted summary table at the end. Report to me:

1. The final test table for all runs (CSI / HSS / SSIM / MSE / LPIPS, plus trainable count).
2. Which surface won, and by how much over the zero-shot floor.
3. How much the 50% and 20% data runs lost relative to the 100% run of the same surface.
4. Any run that failed, with the actual error.

WandB logs to project `Dawncast_foundation` (set `WANDB_STATE=offline` in the sweep script if the
machine has no network).

## Rules

* Do not edit `run_alphapre_convlstm.py`, `models/DAWNCast/dawncast.py`, or
  `finetune_temporal_path_transfer.py` — they are shared with other experiments.
* Do not change hyperparameters (lr 1e-4, batch 4, 50 epochs, seed 0) — the runs must stay
  comparable to the latent-space results.
* If a run fails, do not silently retry with different settings. Report what failed and why.
