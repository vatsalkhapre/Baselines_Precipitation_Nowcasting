# Claude Code prompt — baseline runs

> Paste the block below into Claude Code. Fill the `<<< >>>` placeholders first.

---

## Task

Run precipitation-nowcasting baselines for a paper submission, on 8 GPUs across 3 servers, in 5 days. Everything trains in **full pixel space** (see note on latent below).

**Datasets and horizons:**
| Dataset | T_in → T_out | Resolution |
|---|---|---|
| SEVIR | 5 → 20 | 128×128 |
| MeteoNet | 5 → 20 | 128×128 |
| Shanghai | 5 → 20 | 128×128 |
| CIKM | 5 → 10 | 128×128 |

**Models and current status:**
| Model | Status |
|---|---|
| MAU, SimVP, EarthFormer | Fully done, all 4 datasets |
| PhyDNet | SEVIR done — MeteoNet, Shanghai, CIKM remain |
| TrajGRU | SEVIR done — MeteoNet, Shanghai, CIKM remain |
| ConvLSTM | All 4 remain |
| EarthFarseer | All 4 remain — blocked, see §EarthFarseer |
| AlphaPre | All 4 remain |
| exPreCast | All 4 already run — **pending Audit 3** before accepted as final |
| WADEPre | All 4 already run **on native loss** (correct — no FALFCL substitution per protocol) — **but likely predate the training-rollout fix in §WADEPre and need re-audit before being accepted; may require rerun.** |
| DiffCast | Out of scope this round |

**Model code:** `./models` — one file per model.
**Existing FACL-adapted files:** search `./models` for filenames matching `*_falfcl*` or `*facl*` before writing anything new — do not assume absence, do not ask, just search.
**Paper PDFs:** `./All_Papers` — present on all servers, no need to attach manually.
**Official repo links:** <<< I will give you these >>>
**exPreCast checkpoints:** `/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Exprecast/` — same path mounted on all three servers, kept in sync via GitHub.
**Hyperparameters:** use the argparse defaults in `run_alphapre_convlstm.py`. They are the AlphaPre defaults and are the unified setting for every baseline. Do not invent or tune values.
**Environment:** `earthformer` conda env, on all three servers. Run every job in this env only.
**Reference for evaluation protocol / metric definitions** (CSI-M, pooled CSI, thresholds): the DAWN-Cast paper or the AlphaPre paper, either is authoritative here.
**Multi-GPU:** `run_alphapre_convlstm_multigpu.py` already exists if any single baseline needs more than one GPU. Prefer single-GPU per cell unless a model genuinely requires more.

**Paths:** <<< dataset roots, results CSV path, WandB project >>>

**Note on latent space:** the paper may end up presenting **both** a latent (frozen AutoencoderKL) and a full pixel-space variant of DAWN-Cast itself, framing latent as a compute-saving option for training multiple models. This does **not** change baseline scope — every baseline in this task still trains and is compared in pixel space only. The latent-vs-pixel comparison is a DAWN-Cast-only ablation, out of scope here unless I ask for it separately.

---

## GPU topology — 3 separate servers, not one pool

| Server | GPUs |
|---|---|
| Server A | 3× A6000 (48 GB) |
| Server B | 2× A6000 (48 GB) |
| Server C | 3× A5000 Ada (32 GB) |

**Shared storage confirmed:** dataset paths, `./models`, `./All_Papers`, and the exPreCast checkpoint path are all mounted at the same address on all three servers and kept in sync via GitHub. So a shared run manifest file on this mount is a valid single source of truth across all three — no need to sync data manually.

**Still confirm:** the `earthformer` conda env is present and identical on all three machines before launching anything.

**Live GPU snapshot (will already be stale by the time you read it — poll, don't trust this):**
| Host | Notes |
|---|---|
| `.88` | Running this session. 1 GPU free now; a 2nd freeing soon. |
| `.66` | A5000 Ada server. 1 GPU has an active task (someone else's); rest presumed free. |
| `.205` | 1 GPU has an active task; rest presumed free. |

Map these hostnames to the Server A/B/C groups above as you confirm them — likely `.88`/`.205` are the two A6000 servers and `.66` is the A5000 Ada server, but confirm rather than assume.

**Operational note — this matters for how the dispatcher actually runs:** you (Claude Code) execute locally on whichever machine you're invoked from. This session is on `.88`. To actually run jobs on `.66` and `.205`, the same dispatcher script needs to be started there too — either via a separate Claude Code session on each host, or by having this session `ssh` into them if that's set up. Ask me which before assuming either. All three dispatcher instances coordinate correctly through the shared manifest file since storage is confirmed shared.

**GPU availability changes during the 5 days** — I may reclaim a GPU on short notice (e.g. tomorrow) for other work. This is a single-GPU event on one server, not a full-server loss. The scheduling approach below must handle this without losing progress on the affected run.

---

## Hard rules

1. **Never modify `run_alphapre_convlstm.py` or any other production file.** Create new scripts alongside them.
2. Fix real bugs; do not work around them by adjusting surrounding code.
3. Keep fixed across every baseline: data pipeline, T_in/T_out, resolution, batch size, epoch budget, checkpoint-selection rule.
4. **Checkpoint selection:** validate periodically, keep the checkpoint with the best last-frame CSI-M. Identical rule and interval for every model.
5. All output to stdout. No sweep log files.
6. Report a parameter count for every model before launching its first run. Flag any model wildly out of range of the others.
7. **Do not start any implementation work before completing the audits below.**
8. **Every run must use the `earthformer` conda env. No other env, on any server.**
9. **`log.log` for every run must record, near the top: which conda env is active, and which results CSV path this run writes to.** This is not optional — if two runs are ever compared later and one used the wrong env or wrote to the wrong CSV, that's undetectable without this.
10. **Standing communication rule:** after every discrete task (each audit, each fix, each smoke test, each batch of launches — not just the big checkpoints below), give me a short status update: what's done, what's currently running, what's next. Don't wait to bundle these.
11. **Pre-flight check, before launching anything:** inspect the dataset transform pipelines in `./datasets` and confirm antialiasing is **off** for every resize/transform, consistently across every dataset loader. If any loader has it on while others don't, that silently breaks the "identical data pipeline across all baselines" fairness rule — report it as a finding before touching anything, the same way you would a model deviation.

---

## Audit 1 — paper conformance (do this first, per model)

The paper PDFs for every baseline are attached, and I will give you the official GitHub repo link for each model. Before touching any model file:

1. **Read that model's paper**, specifically the architecture section and the implementation-details / hyperparameter section.
2. **Clone the official repo** (`git clone` into a scratch directory — do not vendor it into our codebase) and read the authors' reference implementation. Where the paper is ambiguous, the authors' code is the tiebreaker; where the two disagree, report the disagreement rather than picking one silently.
3. **Read our corresponding implementation** in this repo.
4. **Compare all three** component by component — module structure, channel dimensions, block counts, normalisation, activation, loss terms and their weights, and anything specified numerically.

Then produce a **conformance report** per model, in this shape:

```
MODEL: <name>
MATCHES:   <components that align with paper + official repo — brief>
DEVIATES:  <component> | paper/repo says X | our code does Y | why it matters
MISSING:   <anything the paper or repo implements that our code does not>
UNCLEAR:   <anything underspecified, or where paper and official repo disagree>
```

**Rules for handling deviations:**

- **Do not silently "fix" the code to match the paper.** Report the deviation first and wait.
- Some deviations are deliberate and correct — e.g. resolution adaptations from 384×384 to 128×128, or the horizon adaptations for same-length models. Distinguish *intentional adaptation* from *implementation error*.
- If a change is genuinely required for the model to run or to preserve the paper's intended behaviour, **make it and then report it explicitly**: what you changed, what the original was, why the change was necessary, and what would have broken without it. One entry per change.
- Never adjust surrounding code to accommodate a bug. Fix the bug or report it.
- The goal is that each baseline is a faithful implementation of its paper, adapted only where our unified protocol demands it — and that every such adaptation is documented well enough to defend in review.

Deliver all conformance reports before starting implementation. I will review them and confirm which changes to proceed with.

---

## Audit 2 — existing FACL model files

**Some FACL/FALFCL-adapted model files already exist in `./models`.** Search for filenames matching `*_falfcl*` or `*facl*` — do not ask me where they are, do not assume they're absent.

For each one found:

1. **Audit it** against both its paper (Audit 1) and the loss protocol below. Check specifically:
   - Is FALFCL substituted in the correct place, and only there?
   - Are native loss terms that should be preserved still intact?
   - Does the `predict(frames_in, frames_gt, compute_loss)` contract match what `run_alphapre_convlstm.py` expects?
   - Are horizons handled correctly for 5→20 and 5→10?
2. **Report what is correct and what is wrong** in each existing file.
3. **Then create a new file** — do not edit or overwrite the existing ones. Carry over what is correct, fix what is not, and list the differences between old and new.

---

## Audit 3 — existing exPreCast results

**exPreCast has already been run on all four datasets and scores exist.** Checkpoints are at `/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Exprecast/` (same path, mounted on all three servers, kept in sync via GitHub). Do not rerun it yet. Verify instead that both the implementation and those completed runs are coherent with the paper.

Check:
1. **Implementation vs. paper + official repo** — full Audit 1 treatment.
2. **Horizon handling** — exPreCast's Temporal Extractor is meant to support unequal horizons natively. Confirm the completed runs actually used 5→20 (SEVIR, MeteoNet, Shanghai) and 5→10 (CIKM), and were not silently padded, truncated, or run same-length.
3. **Resolution** — the paper's SEVIR/MeteoNet configs assume 384×384 and four A6000s. Our runs are 128×128 on the unified defaults. Confirm nothing 384-specific leaked in (patch sizes, window sizes, downsampling factors, hardcoded spatial dims).
4. **Loss** — exPreCast uses FACL natively. Confirm the runs used its own FACL and were not double-wrapped or substituted with our FALFCL.
5. **Checkpoint selection** — confirm the same rule as everything else: validate periodically, keep best last-frame CSI-M. If a different rule or a final-epoch checkpoint was used, the scores are not comparable to the rest of the table.
6. **Metrics** — confirm they were computed by the same evaluation code, at the same thresholds, as every other baseline.
7. **Plausibility** — do the scores sit in a sensible range relative to the completed MAU / SimVP / EarthFormer runs on the same datasets? Flag anything implausibly high or low.

Report findings. **If any check fails, say so and recommend a rerun rather than reusing or patching the numbers.** If everything passes, exPreCast needs no runs and its GPU slot frees up.

---

## Audit 4 — existing WADEPre runs (native loss)

**WADEPre has already been run on all four datasets, on its native loss and curriculum (correct — no FALFCL substitution, per protocol).** However, these runs were very likely produced **before** the training-rollout issue described in §WADEPre below was identified. Do not accept them as final without checking:

1. **Did training supervise all `T_out` frames, or only the first `timesteps` (5) frames** with the remaining frames produced autoregressively at inference only, never trained on? Check the actual training loop that ran, not just the current state of `wadepre.py` — the checkpoint's training log/config is the source of truth for what actually happened.
2. If the runs used **only the first 5 frames supervised** (the flawed original adaptation): these runs are not usable as-is. Flag them for rerun after the §WADEPre fix is implemented. Do not blend partially-fixed and partially-unfixed cells into one results table without saying so explicitly.
3. If the runs already trained through the full rollout with all `T_out` frames supervised: confirm this, and note that §WADEPre's fix is then already satisfied — no rerun needed, just confirm the other §WADEPre checks (`refine_hidden_dim` value used, `self.itr` state, output shape) still hold for these completed runs.
4. Either way, **report which case it is before doing anything else with these runs.**

---

## Audit 5 — adding CSI-181 / CSI-219 to results CSVs

I'd like CSI at intensity thresholds 181 and 219 added to the results CSVs, alongside existing metrics, if this is possible without rerunning training. Treat this as evaluation-only work:

1. **Check whether raw prediction tensors are cached** from completed runs (not just final checkpoints). If so, threshold-based CSI can be recomputed directly from cached predictions — no need to reload models or rerun inference.
2. If raw predictions are not cached, reloading the final checkpoint and rerunning **inference only** (not training) is acceptable to get the predictions needed for the new thresholds.
3. **Before applying this to any completed run, validate on one already-finished baseline first:** recompute an existing metric (e.g. CSI-M or an existing threshold) from scratch using the same code path you'll use for CSI-181/219, and confirm it matches the value already in the CSV. Only proceed to the new thresholds once this matches exactly.
4. **Never overwrite or modify existing CSV rows/columns.** Add CSI-181 and CSI-219 as new columns, appended, for the cells that already have results. Back up the CSV before writing to it.
5. If a cell has no cached predictions and inference is infeasible on the current schedule, report that clearly rather than leaving a silently-blank or fabricated value.

Report which cells got the new columns, which were backed out due to missing predictions, and the validation check from step 3 explicitly.

---

## Loss protocol

| Model | Loss |
|---|---|
| ConvLSTM, TrajGRU, PhyDNet, EarthFarseer | FALFCL |
| AlphaPre | FALFCL on the base regression term only — see below |
| exPreCast | native FACL, unchanged |
| WADEPre | **native loss and curriculum, unchanged** |

### AlphaPre — critical

`self.criterion` is used in **three** places in `predict()`. Substitute FALFCL in **only the first**:

```python
loss += self.criterion(pred, frames_gt)      # ← FALFCL HERE, and nowhere else
loss += self.pha_weight  * pha_loss          # keep native
loss += self.amp_weight  * amp_loss          # criterion(xas_abs, frames_abs) — keep native
loss += self.anet_weight * anet_loss         # criterion(xas, frames_gt)      — keep native
```

Globally replacing `self.criterion` would apply FALFCL to FFT magnitude tensors in `amp_loss`. That is meaningless and will error or silently produce garbage.

`amp_weight` has its own decay schedule that now coexists with FALFCL. Log both weights every validation step.

---

## Per-model work

### WADEPre — 🔴 fix before running, contingent on Audit 4

`wadepre.py` is a same-length model (T_in → T_in) adapted with an autoregressive rollout. As written:
- **Training** supervises only `frames_gt[:, :timesteps]` — lead times 1–5.
- **Inference** rolls out 4× — lead times 6–20 come from the model's own predictions.

15 of 20 output frames are never supervised and are produced from a distribution never seen in training. This will collapse WADEPre's long-lead CSI and is not a fair baseline.

**Required change:** run the same `n_rollout` loop at training time and supervise **all `T_out` frames**.
- Backprop through all rollout steps.
- If it OOMs on a 48 GB A6000 at 128²: gradient-checkpoint across rollout steps first. Detaching between blocks is the last resort — report clearly if you use it.
- CIKM has `n_rollout=2` — apply the same fix.
- **If Audit 4 finds the existing 4 runs already used this full-horizon supervision, this fix is already satisfied for them — no rerun needed.** If Audit 4 finds only the first 5 frames were supervised, all 4 existing WADEPre runs need to be redone with this fix, which changes the run count and schedule accordingly.

**Also fix / verify:**
- `get_model` defaults `refine_hidden_dim = timesteps * 96` = **480** at T_in=5, but the comment above it claims 560 via `_safe_refine_hidden_dim` (which only runs when a value is passed explicitly). Paper value is 576. Pick one, log it.
- `self.itr` is a plain int outside the state dict → resuming a checkpoint restarts the `loss_a` curriculum from step 0. Persist it, or forbid resume.
- `predict(compute_loss=True)` returns `(B, 5, 1, H, W)` against `frames_gt` of `(B, 20, 1, H, W)`. Check the runner does not compute train-time metrics on this. The rollout fix resolves it.

### EarthFarseer — 🔴 blocker, needs implementation

`earthfarseer.py` is **hard-locked to T_in = T_out**. `forward()` ends:

```python
predictions = predictions.reshape(B, T, C, H, W) + skip_feature
```

`T` is the input length throughout, and the `+ skip_feature` residual (from `skip_connection`, itself T→T) ties output length to input length. The paper describes a **Temporal Projection** decoder stage (Eq. 9) mapping `T×C → K×C` for arbitrary `K` — **it is not in the released code.**

**Preferred fix:** implement the paper's Temporal Projection — a `ConvNormRelu` over concatenated `T*C` channels → `K*C`, applied after `self.dec`. The `+ skip_feature` residual must be projected too, or dropped when `K ≠ T`. Document it as a reimplementation of the authors' described design.
**Fallback:** autoregressive rollout as with WADEPre, with the same train-through-rollout requirement.

**Also fix:**
- `self.enc = Encoder(C, hid_S, N_S)` is constructed and **never used** in `forward`. With `hid_S=512` this is a large block of dead parameters. Remove it before reporting parameter counts.
- The local `Mlp.forward` never calls `self.fc2` — allocated, unused.
- `hid_S=512` feeds `TeDev(T*hid_S, ...)` = 2560 channels at T=5. Report the parameter count and compare against the other baselines before launching.
- `skip_connection` is built with its own defaults (`hid_S=16`), ignoring the outer `hid_S=512`. Confirm this is intended.
- Needs a `predict(frames_in, frames_gt, compute_loss)` wrapper matching the runner contract (see `wadepre.py` for the pattern). No loss in the model — FALFCL wraps cleanly.
- Imports are flat (`from FoTF_module import *`, `Temporal_block`, `utils`) — package them properly.
- `H1/W1` resolve cleanly at 128 (`128 % 3 != 0` → 32). No action needed.

### exPreCast
**Already run on all four datasets — do not rerun until Audit 3 is complete.** Already uses FACL natively, so no loss substitution. Its paper config assumes 384×384 and 4 GPUs; do not port it. If Audit 3 fails on any dataset, rerun only the affected cells on the unified defaults at 128².

### ConvLSTM, TrajGRU, PhyDNet
Straightforward. Unified defaults + FALFCL. TrajGRU and PhyDNet must each match the settings of their already-completed SEVIR run exactly — only MeteoNet, Shanghai, CIKM remain for both.

### AlphaPre
Lowest integration cost — the runner is already built around it. Apply the loss rule above.

---

## Execution — job queue + dispatcher, not a fixed GPU table

GPU availability changes during the run (see Topology section) — do not write a script that assumes a fixed GPU stays assigned to one model for 5 days straight. Build instead:

### 1. A run manifest — single source of truth

One file (CSV or JSON), listing every cell:

```
model, dataset, server, status, gpu_index, checkpoint_path, config_hash, last_updated
```

`status` ∈ `{pending, running, done, failed}`. Update it atomically (write to a temp file, `os.rename` over the original) so two dispatcher instances never claim the same cell simultaneously.

### 2. Cell → server assignment (fixed; GPU within a server is dynamic)

Assign by where the dataset actually lives (confirm per Topology §1):

| Server | Cells |
|---|---|
| Server A (3× A6000) | WADEPre × 0–4 (contingent on Audit 4), EarthFarseer × 4 |
| Server B (2× A6000) | AlphaPre × 4, exPreCast reruns (only if Audit 3 fails) |
| Server C (3× A5000 Ada) | ConvLSTM × 4, TrajGRU × 3 (MeteoNet/Shanghai/CIKM), PhyDNet × 3 (MeteoNet/Shanghai/CIKM) |

Within a server, do not hand-pin a cell to a specific GPU index.

### 3. A per-server dispatcher script

- Polls free GPU indices on that machine — via `nvidia-smi` and/or a `reserved_gpus.txt` I can edit by hand the moment I grab a card for something else.
- Launches the next `pending` cell assigned to that server on whichever GPU is free, with `CUDA_VISIBLE_DEVICES` set **before** Runner instantiation.
- On launch, marks the cell `running` with its GPU index; on completion, `done`; on crash/kill/preemption, back to `pending` (keeping its last checkpoint path so it resumes rather than restarts).
- Runs the sanity gate automatically at ~10% of that cell's step budget — NaN check, loss divergence, non-trivial CSI. On failure: log it, mark the cell `failed`, do not let it silently burn its remaining budget.

### 4. Checkpoint-resumability is a hard requirement, not a nice-to-have

Because GPU availability is dynamic, **every model must resume correctly from a mid-training checkpoint**, since preemption is now a routine event, not a rare one. Before launching anything for real:

- Fix WADEPre's `self.itr` (§ WADEPre section above) — it's a plain int outside the state dict, so a naive resume silently restarts the `loss_a` curriculum from step 0 while training-step count moves on, desyncing the schedule from actual progress. Persist it in the checkpoint.
- Verify the same for every other model with a step-dependent schedule (AlphaPre's `amp_weight` decay, any LR warmup/scheduler state). If the optimizer/scheduler state isn't in the checkpoint, add it.
- Test resumability explicitly on the CIKM smoke test for every model before trusting it on a full run: kill a run partway through, resume it, and confirm the loss curve continues smoothly rather than jumping.

### 5. Order per model, wherever it lands

CIKM (smoke test) → SEVIR (headline) → Shanghai → MeteoNet.

### 6. Final verification pass — before any cell counts as done

After a cell reaches `done`: reload its final checkpoint, recompute its metrics independently, and confirm the recomputed number matches what's in the manifest/results CSV. Confirm frame count, non-NaN, and correct horizon. **A cell only counts as complete once this passes** — there is no budget to discover a bad number after the fact.

`nargs='+'` argparse args take space-separated values, not quoted strings.

---

## Order of work

1. **Confirm the `earthformer` env is present and identical on all 3 servers**, and confirm/correct the hostname→server mapping in the Topology section.
2. **Audit 1** — paper conformance for all models, cross-checked against the official repos I give you and papers in `./All_Papers`. Deliver every conformance report. Stop and wait for my review.
3. **Audit 2** — search `./models` for existing `*_falfcl*`/`*facl*` files, audit them, then write new files (never overwrite).
4. **Audit 3** — verify the existing exPreCast implementation and its four completed runs. Report pass/fail per dataset.
5. **Audit 4** — verify whether the existing WADEPre native-loss runs supervised all `T_out` frames or only the first 5. Report which case it is before proceeding.
6. **Audit 5** — assess feasibility of adding CSI-181/CSI-219 to existing results CSVs without retraining. Validate against one existing metric before touching any real CSV.
7. **Then, in parallel:** the WADEPre training-rollout fix (informed by Audit 4's finding) and the EarthFarseer temporal-projection implementation. Nothing else runs until these are done.
8. Write the AlphaPre FALFCL wrapper (first `self.criterion` only).
9. **Fix resumability** for every model with step-dependent state (WADEPre's `self.itr`, AlphaPre's `amp_weight` schedule, any scheduler/optimizer state) — required given dynamic GPU availability.
10. Build the run manifest and the per-server dispatcher scripts.
11. Smoke-test every model on CIKM at a tiny step count, **including an explicit kill-and-resume test**, before committing any full run.
12. Launch via the dispatchers.
13. Run the final verification pass on every cell before treating any number as final.

Report parameter counts and CIKM smoke-test + resume-test results before starting the full schedule.

**Checkpoints where you stop and wait for me:** after step 1, after step 2, after step 3, after step 4, after step 5, after step 6, and after step 11.

**Remember the standing rule:** a short done/ongoing/next update after every discrete task in this list, not only at these checkpoints.
