# Baseline Run Plan — DAWN-Cast (ICLR26) · v3

**Budget:** 5 days · 3 servers: 3× A6000 (Server A) + 2× A6000 (Server B) + 3× A5000 Ada (Server C) = 8 GPUs, shared storage confirmed across all three
**Datasets (4):** SEVIR, MeteoNet, Shanghai, CIKM — all at **128×128**
**Training space:** baselines run in **full pixel space**. DAWN-Cast itself may present **both** a latent (frozen AutoencoderKL) and pixel-space variant in the paper — framed as a compute-saving option for training multiple models. This is a DAWN-Cast-only ablation and does not change baseline scope.
**Loss:** FALFCL for all runnable baselines except exPreCast (native FACL) and WADEPre (native curriculum)
**Protocol:** 1 run per (model, dataset) cell
**Environment:** `earthformer` conda env only, on all servers

---

## 1. Run matrix

| Model | SEVIR | MeteoNet | Shanghai | CIKM | Runs remaining |
|---|---|---|---|---|---|
| ConvLSTM | ☐ | ☐ | ☐ | ☐ | 4 |
| TrajGRU | ✅ | ☐ | ☐ | ☐ | 3 |
| PhyDNet | ✅ | ☐ | ☐ | ☐ | 3 |
| MAU | ✅ | ✅ | ✅ | ✅ | 0 |
| SimVP | ✅ | ✅ | ✅ | ✅ | 0 |
| EarthFormer | ✅ | ✅ | ✅ | ✅ | 0 |
| EarthFarseer | ☐ | ☐ | ☐ | ☐ | 4 |
| AlphaPre | ☐ | ☐ | ☐ | ☐ | 4 |
| exPreCast | ✅* | ✅* | ✅* | ✅* | 0, pending Audit 3 |
| WADEPre | ✅† | ✅† | ✅† | ✅† | 0, pending Audit 4 |
| ~~DiffCast~~ | — | — | — | — | deferred |

\* exPreCast: all 4 done, but not yet verified against paper/repo/protocol (Audit 3).
† WADEPre: all 4 done **on native loss (correct choice)**, but very likely predate the training-rollout fix in §8.1 below — i.e. may have supervised only 5 of 20 output frames. **Must be re-audited (Audit 4) before being accepted; may require full rerun of all 4.**

**Firm remaining runs: 18** (ConvLSTM 4 + TrajGRU 3 + PhyDNet 3 + EarthFarseer 4 + AlphaPre 4). **Conditional: 0–8 more** depending on Audit 3 (exPreCast) and Audit 4 (WADEPre) outcomes.

---

## 2. ⚠️ The WADEPre differentiation got weaker when you dropped latent

Your rebuttal argument previously had three legs. Retiring the latent path removed one, and you may not have re-checked what's left — though note the latent path is now back on the table for DAWN-Cast itself (see header), just not shared with WADEPre's usage of it.

**Both models are now: pixel-space, DWT decomposition, separate treatment of approximation vs detail subbands, with a loss curriculum.** That is a much tighter overlap than it was.

**What your current text argues:**
1. WADEPre is fixed 6→6; DAWN-Cast supports arbitrary horizons.
2. WADEPre uses DWT as a *representation* mechanism (two networks + refiner); DAWN-Cast uses it to enable *distinct temporal modelling per subband* — an independent FAT block per subband rather than one shared temporal path. **Confirmed: the FAT block is Gabor + MLP** (dual-stream), operating on the temporal axis — this is the concrete mechanism to name explicitly rather than leaving it as "an independent block per subband."

**Honest assessment:**

- **Argument 1 is your weaker one and you led with it.** Flexible horizon is a capability claim, not an architectural novelty claim. A reviewer can respond "WADEPre's Refiner could be extended to unequal horizons in an afternoon." Keep it as a practical advantage; don't rest novelty on it.

- **Argument 2 is the real one and it's under-sold.** The rebuttal text never mentions **Gabor**. The FAT block's actual distinguishing mechanism is a GaborLayer operating along the *temporal* axis with per-subband frequency adaptation — that is not something WADEPre has any analogue of. Right now you're describing FAT as "an independent block per subband," which reads as *architectural bookkeeping* (they have 2 branches, we have 4). Name the mechanism. The wavelet-statistics motivation (Fig. 5 / Appendix C) is doing the right work — pair it with Gabor explicitly.

- **"across the majority of forecasting metrics"** reads to a reviewer as "loses on some." If that's true, say which and why (e.g. WADEPre wins on a sharpness metric but loses CSI at extreme thresholds) — owning it is stronger than hedging.

- **Terminology:** your text uses "sharing a single temporal *operator*." Your own house rule bans "operator." Use "temporal path" or "temporal module."

- **Timing kills the concurrent-work framing.** WADEPre is Feb/Mar 2026; you submit ~Sept 2026. That is ~6 months of prior art, not concurrent work. The line *"Although this does not affect the proposed methodology, we will include a brief discussion of wavelet based methods... for completeness"* was acceptable in a rebuttal but is **too dismissive for the new submission**. WADEPre needs a proper related-work paragraph and a table row, not a completeness footnote. Reviewers who know the area will check whether you engaged seriously.

**Practical consequence for the runs:** WADEPre is architected for equal input/output horizons. If any of your four datasets uses unequal T_in→T_out, you must adapt it — and you must document exactly how, because that adaptation is simultaneously (a) an integration risk and (b) the evidence for your own horizon-flexibility claim. Do WADEPre first.

---

## 3. Per-model configuration

| Model | Config source | Loss | Notes |
|---|---|---|---|
| ConvLSTM | **Your defaults** | FALFCL | No dataset overlap. Clean. |
| TrajGRU | **Your defaults** | FALFCL | Shi et al. 2017; no 128² config for your datasets. |
| PhyDNet | Your defaults (as used for SEVIR) | FALFCL | Keep identical to your completed SEVIR run. |
| EarthFarseer | **Your defaults** — see ⚠️ below | FALFCL | Paper trains with MSE → Category 1, safe. |
| AlphaPre | Paper config (matches yours) | see ⚠️ | bs 4, lr 1e-4, Adam, 100 ep, α=0.1, β=0.01, Np=3, Na=3, trunc. freq 20 |
| exPreCast | **Your defaults** — see ⚠️ below | **native FACL** | Already uses FACL — no substitution needed. |
| WADEPre | Paper config, adapted | see ⚠️ | l=3, bior2.4; Approx 256ch/3 STBlocks; Details base 128, FPN [64,128,256], IDR 64; Refiner 576; AdamW lr 1.5e-4, β=(0.9,0.995), wd 0.01, fp32, CosineAnnealingLR T_max=200 |

### ⚠️ EarthFarseer — the "SEVIR match" does not actually transfer

Two problems with treating `bs=16, 300 epochs, lr=0.01, Adam` as an official config:

1. **Their SEVIR is 384×384, 10→10** (paper Table 1). You are at 128×128. Batch size and LR do not carry across a 9× change in pixel count.
2. **That paragraph is a scalability ablation**, not the main experimental setup. It describes a sweep over 2–14 TeDev blocks to show block-count scaling — not the config that produced the Table 2 headline numbers. Citing it as "the paper's config" is not accurate.

Also: **lr=0.01 with Adam is atypically high** for this model class. And 300 epochs on SEVIR will not fit a 30 h slot.

→ **Use your defaults.** Set ST block count explicitly (paper defines Ti/S/B = 6/12/24 blocks; pick one and record it). State in the paper that EarthFarseer's published config was for a different resolution and that you used the unified pipeline.

### ⚠️ exPreCast — resolution mismatch confirmed

Their SEVIR and MeteoNet configs assume 384×384 and need 4× A6000. At 128² neither the batch size nor the multi-GPU setup transfers. Their schedule (100 epochs SEVIR; 100k iters @ bs 16 elsewhere; validate every 5k iters on last-frame CSI-M, keep best) is a reasonable *anchor* but not a drop-in config.

→ **Use your defaults across all four cells.** Their best-checkpoint selection rule (validate periodically, keep best CSI-M) is worth adopting **uniformly across every baseline including yours** — it is a fairness lever reviewers notice.

### ⚠️ AlphaPre — batch size 4

Paper config is `bs=4`. If your unified pipeline uses a different batch size, you have a conflict with the "batch size fixed across all baselines" fairness rule. Pick one and be explicit:
- Use your unified batch size → deviates from paper, but fair across the table. **Recommended.**
- Use bs=4 → matches paper, breaks uniformity, and AlphaPre gets more gradient steps per epoch than everyone else.

Loss composition still pending from you (pure regression + hybrid terms with α=0.1, β=0.01).

---

## 4. Loss protocol

| Category | Models | Action |
|---|---|---|
| 1 — pure/deterministic regression | ConvLSTM, TrajGRU, PhyDNet, EarthFarseer, AlphaPre* | Substitute FALFCL for the regression term |
| Native FACL | exPreCast | No change — already FACL |
| Mechanism-critical | WADEPre | See below |
| Deferred | DiffCast | Not run this round |

\* pending confirmation of AlphaPre's hybrid terms.

**WADEPre decision required.** Its dynamic weight annealing (λ_D=0.05, λ_Mixed=0.005, T_decay=3000 steps, λ_min=0.01) is a coarse-to-fine curriculum — i.e. the component most similar to FALFCL. Two options:
- **Run with native annealing.** Beating an intact WADEPre is the result that matters, and swapping its curriculum for yours invites "you ablated the part that competes with your contribution."
- **Run both** (native + FALFCL-substituted) if the spare GPU allows. Strongest possible position.

**The paper must state, per baseline, whether the loss was native or FALFCL-substituted.** One blanket sentence will not survive review.

---

## 5. GPU topology and dynamic scheduling

### 5.1 Physical layout — 3 servers, not one shared pool

| Server | GPUs | Notes |
|---|---|---|
| Server A | 3× A6000 (48 GB) | |
| Server B | 2× A6000 (48 GB) | |
| Server C | 3× A5000 Ada (32 GB) | |

**This is not one interchangeable 8-GPU pool.** A static "GPU #4 runs EarthFarseer" plan breaks the moment a machine has a different data path, conda env, or you reclaim a card on it mid-run. Two things must be confirmed before any job placement is finalized:

1. **Do all three servers see the same dataset storage and results CSV path** (shared NFS/mount), or does each have its own local copy? If local, code + data must be synced to each server first, and a job can only run on the server where its dataset actually lives.
2. **Is the conda env (`<<< env name >>>`) identical and present on all three machines?** If not, resolve that before scheduling, not after a job fails on server B.

Given you said you may reclaim a GPU "tomorrow, on the way" — this is almost certainly **within one of these servers, not a full-machine loss**. Design for per-GPU reclaim, not per-server.

### 5.2 Why static assignment is the wrong model here

A fixed table ("this GPU runs this model for 5 days straight") assumes GPU availability is constant. You've said it isn't. If a static plan loses a GPU mid-run with no resume path, that cell's partial progress is wasted — and **you don't have the runway to redo a lost cell.**

**Replace the static table with a job queue + dispatcher, per server:**

- **A run manifest** — one CSV/JSON file, the single source of truth, listing every (model, dataset) cell with status: `pending / running / done / failed`, which GPU/server it's on, its checkpoint path, and its resolved config. Update it atomically (write to a temp file, then rename) so two dispatchers never double-claim a cell.
- **A per-server dispatcher script** — polls `nvidia-smi` (or a simple `reserved_gpus.txt` you edit by hand when you grab a card) for free GPU indices on that machine, and launches the next `pending` job from the manifest that belongs on that server, wrapped with the correct `CUDA_VISIBLE_DEVICES`.
- **Every job is checkpoint-resumable.** If a GPU disappears mid-run (you reclaim it, or it's preempted), the dispatcher marks that cell `pending` again with its last checkpoint path, and it resumes — on the same GPU later, or a different free one — instead of restarting from step 0.

This turns "I might grab a GPU tomorrow" from a scheduling emergency into: the dispatcher just sees one less free GPU and queues around it.

### 5.3 Cell-to-server assignment (fixed; GPU-within-server is dynamic)

Assign cells to servers based on where their data lives (§5.1, question 1). Suggested grouping, adjust once storage is confirmed:

| Server | Cells assigned |
|---|---|
| Server A (3× A6000) | WADEPre × 4, EarthFarseer × 4 |
| Server B (2× A6000) | exPreCast reruns (if Audit 3 fails), AlphaPre × 4 |
| Server C (3× A5000 Ada) | ConvLSTM × 4, TrajGRU × 4, PhyDNet × 3 |

Within each server, the dispatcher fills whichever GPUs are actually free — do not hand-pin a cell to "A6000 #2" specifically.

**Order per model, wherever it runs:** CIKM (smoke test) → SEVIR (headline) → Shanghai → MeteoNet.

### 5.4 Correctness guarantees — mandatory, no exceptions

You said there is no second chance. That means every run needs to be independently verifiable after the fact, not just "it finished":

1. **Sanity gate at ~10% of budget** (unchanged): NaN check, loss divergence, non-trivial CSI. A diverged run is caught at hour 3, not discovered at hour 28.
2. **Log the fully-resolved config with every checkpoint** — every hyperparameter, which loss variant was used (native vs. FALFCL-substituted, and exactly where), horizon-adaptation method, git commit hash of the model file. If a number is questioned in review six weeks from now, you need to reconstruct exactly what produced it without guessing.
3. **Fix resumability bugs before relying on resume.** WADEPre's `self.itr` is a plain int outside the state dict (§8.4) — resuming restarts the loss-curriculum schedule from step 0 and silently desyncs it from training progress. Under a dynamic dispatcher, resume is now a **routine** operation, not a rare edge case, so this must be fixed for every model before launch, not discovered after a preemption corrupts a run.
4. **Final verification pass, after every cell is marked `done`:** reload each final checkpoint, recompute its metrics independently, and confirm the number in the manifest matches. Confirm no cell has a truncated frame count, a NaN metric, or an evaluation run with the wrong horizon.
5. **No cell is `done` until it passes checks 1 and 4.** If a check fails, the cell goes back to `pending` and reruns using whatever slack capacity exists — do not report a number you haven't verified.

**Integration is still the binding constraint, not GPU hours.** Front-load all adapter/wrapper work (WADEPre rollout fix, EarthFarseer temporal projection, AlphaPre loss wrapper, TrajGRU integration) in parallel on day 1, before the dispatcher has anything correct to run.

---

## 6. Still needed before the Claude Code prompt

1. **Your default hyperparameters** — LR, batch size, optimizer, epochs/iters, scheduler, warmup. Never actually stated in this thread; the prompt cannot be written without them.
2. **T_in → T_out per dataset** (SEVIR, MeteoNet, Shanghai, CIKM). Determines whether WADEPre needs horizon adaptation.
3. **AlphaPre's hybrid loss composition** (you said next prompt).
4. **Which WADEPre cells already exist**, and under which loss.
5. **EarthFarseer ST block count** — Ti (6), S (12), or B (24)?
6. **Checkpoint selection rule** — adopting exPreCast's "validate periodically, keep best CSI-M" uniformly? If yes, at what interval?
7. **Results sink** — CSV path + WandB project/naming convention for these runs.

---

## 7. Deferred

- **DiffCast** — Category 2 loss, incompatible with FALFCL substitution; also likely requires a per-dataset pretrained backbone (2 trainings per cell). Out of scope this round.
- **CasCast, NowcastNet** — add only if time allows after the 27 runs land.

---

## 8. Code-level findings (v3 — from `wadepre.py` and `earthfarseer.py`)

Horizons: **T_in=5 → T_out=20** for SEVIR / MeteoNet / Shanghai; **5 → 10** for CIKM.
Two of the three competitive baselines are same-length models. How you adapt them is now the single largest fairness risk in the paper.

### 8.1 🔴 WADEPre: 15 of 20 output frames are never supervised

Current `predict()`:
- **Train:** single step, `truth = frames_gt[:, :self.timesteps]` → loss on lead times **1–5 only**.
- **Infer:** 4 autoregressive rollouts → lead times **6–20 produced from its own predictions**.

Frames 6–20 are (a) never supervised and (b) generated from an input distribution the model never saw in training. Textbook exposure bias. WADEPre's CSI will collapse over exactly the long-lead region where your paper claims its win.

**This is the "you crippled the closest competitor" attack, pre-built.** WADEPre is prior art from ~6 months before your submission, it is the model a reviewer will scrutinise hardest, and 75% of its output is untrained.

**Fix — train through the rollout.** Run the same `n_rollout` loop at training time and supervise all `T_out` frames:
- 4 forward passes, backprop through all of them; ~4× activation memory.
- If it OOMs on A6000 at 128²: gradient-checkpoint across rollout steps first. Detaching between blocks is the last resort — it restores supervision on all frames but leaves a residual train/test gap, and must be stated in the paper.
- CIKM (`n_rollout=2`) is half as severe but needs the same treatment.

Whatever you choose, **the adaptation must be described explicitly in the paper.** It is simultaneously the fairness argument and the evidence for your own horizon-flexibility claim.

### 8.2 🔴 EarthFarseer: the released model is hard-locked to T_in = T_out

The paper describes a two-stage decoder whose second stage is a **Temporal Projection** (Eq. 9) mapping `T×C → K×C` for arbitrary `K`. **`earthfarseer.py` does not implement it.** `forward()` ends:

```python
predictions = predictions.reshape(B, T, C, H, W) + skip_feature
```

`T` is the *input* length throughout, and the residual `+ skip_feature` (from `skip_connection`, itself T→T) hard-ties output length to input length. This is why the earlier attempt failed — there is nothing to configure.

**Options, in order of preference:**
1. **Implement the paper's Temporal Projection.** A `ConvNormRelu` over the concatenated `T*C` channels → `K*C`, applied after `self.dec`. The `+ skip_feature` residual must also be projected, or dropped when `K ≠ T`. Defensible (you are implementing the authors' described design) but must be documented as a reimplementation.
2. **Autoregressive rollout**, matching WADEPre — with the same train-through-rollout requirement from §8.1.
3. Build with `T = T_out` and pad the 5 real input frames. Wasteful and distorts the temporal encoder.

**Other EarthFarseer landmines:**
- `self.enc = Encoder(C, hid_S, N_S)` is constructed in `__init__` and **never used** in `forward`. With `hid_S=512` this is a large block of dead parameters that will inflate any reported parameter count. Remove before reporting params.
- The file's local `Mlp.forward` never calls `self.fc2` — allocated, unused.
- `hid_S=512` feeds `TeDev(T*hid_S, ...)` = 2560 input channels at T=5. Check the parameter count against your other baselines before launching; this may be far outside the range of the rest of the table.
- Note `skip_connection` is built with its own defaults (`hid_S=16`), ignoring the outer `hid_S=512`.
- `H1/W1` resolve cleanly at 128 (`128 % 3 != 0` → `H1 = W1 = 32`). No issue at your resolution.
- No `predict()` interface and no loss — needs a wrapper matching the runner contract. FALFCL wraps cleanly (Category 1 confirmed).

### 8.3 ⚠️ AlphaPre: `self.criterion` appears three times — substitute only the first

```python
loss += self.criterion(pred, frames_gt)      # ← FALFCL goes HERE, and only here
loss += self.pha_weight  * pha_loss          # keep native
loss += self.amp_weight  * amp_loss          # criterion(xas_abs, frames_abs) — keep native
loss += self.anet_weight * anet_loss         # criterion(xas, frames_gt)     — keep native
```

Globally swapping `self.criterion` would apply FALFCL to **FFT magnitude tensors** in `amp_loss`, which is meaningless and will likely error or silently produce garbage.

Also: the code has **four** loss terms; the paper's equation has three (MSE + α·Amplitude + β·Phase). `anet_loss` is not in the published equation. Describe what you ran, not what the paper wrote.

`amp_weight` decays to 0 on its own schedule — a curriculum that now coexists with FALFCL. Watch for interaction and log both weights.

### 8.4 ⚠️ WADEPre code nits
- `get_model` default: `refine_hidden_dim = timesteps * 96` = **480** at T_in=5, but the comment directly above claims 560 via `_safe_refine_hidden_dim`. That helper only runs when a value is explicitly passed. Paper value is 576. Pick one deliberately and record it.
- `self.itr` is a plain int outside the state dict → **resuming from a checkpoint restarts the `loss_a` curriculum from step 0**. With `T_decay=3000` and best-checkpoint selection, any resume desyncs the schedule.
- `predict(compute_loss=True)` returns `(B, 5, 1, H, W)` while `frames_gt` is `(B, 20, 1, H, W)`. If the runner computes train-time metrics on the returned `pred`, this will shape-error. (Resolved automatically if §8.1's fix is applied.)
- WADEPre keeps its **native** loss and curriculum — no FALFCL substitution. This is the deliberate choice from §4.

### 8.5 Checkpoint selection
Confirmed: validate periodically, keep best last-frame CSI-M. Applied uniformly to every baseline **and** DAWN-Cast. Record the interval in the paper.

---

## 9. Session 3 updates — operational (full detail in the Claude Code prompt)

- **exPreCast checkpoints:** `/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Exprecast/`, same path mounted on all 3 servers, synced via GitHub.
- **Papers:** `./All_Papers` on every server — no manual PDF attachment needed going forward.
- **Model code:** `./models`. Existing FACL-adapted files: search for `*_falfcl*`/`*facl*` rather than asking.
- **Environment:** `earthformer` conda env, all servers, no exceptions.
- **Live GPU snapshot at time of writing** (will be stale — dispatcher must poll, not trust this): `.88` (this session) 1 free now + 1 freeing soon; `.66` (A5000 Ada server) 1 GPU busy elsewhere; `.205` 1 GPU busy elsewhere. Hostname→Server A/B/C mapping still needs confirming.
- **Multi-GPU:** `run_alphapre_convlstm_multigpu.py` already exists if any single cell needs more than one GPU.
- **New pre-flight check:** confirm antialiasing is off across every dataset transform in `./datasets`, consistently. An inconsistency here silently breaks the fixed-data-pipeline fairness rule the same way a per-model config drift would.
- **New logging requirement:** every run's `log.log` records the active conda env and the results CSV path it writes to.
- **New metrics request (Audit 5):** add CSI-181 and CSI-219 to results CSVs where feasible from cached predictions or checkpoint-reload inference only — never by retraining, never by overwriting existing columns, and only after validating the recomputation pipeline against one already-known-correct metric.
- **Standing rule:** status update (done / ongoing / next) after every discrete task, not only at major checkpoints.
