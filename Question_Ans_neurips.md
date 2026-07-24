# DAWN-Cast — NeurIPS Rebuttal Draft

> **Reviewing guidance (from the conference):** *"Use the initial meta-review as your guide. It tells you what would most likely change the AC's view of the paper. Focus your response on those points."*
> → The **AC's points and weaknesses are the priority**; the potentially decision-changing items are called out below.

> **Legend — plots referenced in this draft**
> - **Sinusoidal-frequency plot** (referred to below as *"graph"* / *"frame"*): x-axis = sinusoidal frequency, y-axis = low- vs high-frequency wavelet behavior. It shows that specific Gabor parameters in the FAT Block replicate large-scale and small-scale turbulence-structure motion for the region a dataset is drawn from, and to some extent justifies the **explicit** multiscale wavelet decomposition.
> - **CSI–MSE plot**: used to show the correlation between CSI and MSE (why threshold-based metrics improve while MSE does not).

> **⚙️ TO-RUN items are marked with `▶️ RUN` callouts throughout, and consolidated in the [Action Items](#action-items) checklist at the end so none are missed.**

---

# AC

**Weaknesses:**

### 1. Limited conceptual novelty

> The main concern is limited conceptual novelty. Wavelet decomposition, spectral modeling, multi-scale processing, Gabor activations, and global-local refinement are all established ideas. The paper does not clearly identify which component constitutes the main methodological advance beyond their integration.

**A.** Main contribution: the Gabor activations are adaptable to a particular environmental location. A **separate FAT Block for each separately wavelet-decomposed band**, with each Gabor adapting to those environmental characteristics, is our novelty. As we see in the sinusoidal-frequency plot, running different sinusoidal patterns for the low and high wavelet frequencies settles at one set of sinusoidal behavior for a given frequency and not another — meaning that modeling a particular wavelet frequency with this particular sinusoidal frequency gives optimal outputs. This leads us to the conclusion that sinusoids are accurately able to model different wavelet frequencies, leading to SOTA results, and implying that the wavelet decomposition and separate Gabor blocks for those wavelets hold significance.

One could argue that there is not much score difference; this argument can be countered by the fact that the FAT Block has very few parameters compared to the refinement block of the model, so this is expected. Hence in the paper we say in the limitations section (lines 462–467): *"The learnable parameters of the FAT blocks constitute approximately minor percentage of the total model parameter count. Consequently, the per-dataset ablation score deltas attributable to the adaptive component alone are modest."* and in the future work section (lines 319–320): *"scaling the adaptive part to a larger fraction of model capacity to assess its full potential."* We accept this, but this framework presents a future direction for an adaptive model **with SOTA results**. So the paper actually acts as motivation for how future precipitation-nowcasting models can adopt a better adaptive approach in order to achieve better precipitation nowcasting in the near future.

### 2. Ablation weakens the central claim (SRST vs Gabor)

> The ablation results also weaken the central claim. Removing the SRST block causes the largest performance drop, although this block mainly combines existing AFNO-style processing with depthwise convolution. In contrast, removing the Gabor stream produces only modest degradation. This suggests that the main gains may not come from the proposed frequency-adaptive temporal mechanism.

**A.** Again, the two limitations/future-work arguments cited above apply — limitations section (lines 462–467): *"The learnable parameters of the FAT blocks constitute approximately minor percentage of the total model parameter count. Consequently, the per-dataset ablation score deltas attributable to the adaptive component alone are modest."* and future work section (lines 319–320): *"scaling the adaptive part to a larger fraction of model capacity to assess its full potential."* We accept this, but this framework presents a future direction for an adaptive model.

We could have presented just this as the motivation, but without convincing results it would be difficult for the community to accept the idea. Hence we added some of the conventional spatio-temporal forecasting parts, like global–local refinement, in order to achieve better scores and eventually improve the chances of the idea's acceptability.

### 3. Physical interpretation and "climatic inductive bias"

> The physical interpretation is not sufficiently supported because wavelet decomposition is applied to learned latent features rather than directly to radar fields. It is therefore unclear whether the resulting subbands correspond to physically meaningful precipitation scales. The "climatic inductive bias" claim is also overstated, since the model does not use climate labels or environmental variables.

**A.** We use the autoencoder for the latent → physical-data correspondence proof from *"Your Latent Mask is Wrong,"* arXiv:2512.05198, Dec 2025.

> ▶️ **RUN / CHECK:** Verify the latent→physical correspondence argument against arXiv:2512.05198; also review the Claude response on this question as it is worth exploring.

We say *climatic* inductive bias because a particular dataset is taken from a particular location, and every location has a certain climate. Gabor application on the wavelet-decomposed data (corresponding to a bulk phenomenon and a turbulence phenomenon) means that only one Gabor configuration can model each — highlighting that the outputs from the FAT Blocks are highly usable by the remaining refinement block to produce even better predictions. That initial meaningful output from the FAT Block provides the inductive bias the remaining model needs for a better idea of the climatic evolution in the locality. As radar images are nothing but the shadow of precipitation bands, and precipitation bands are directly affected by the climate variables of the region, this indirectly implies that the movement and evolution of radar images are indirectly affected by the climate of the region. Hence we call it an inductive bias.

> **Note (terminology):** the paper already uses *"climatic inductive bias"* / *"climatically-aware inductive bias"* (Abstract, Sec. 2.1.1). Reviewer 3 suggests *"frequency-adaptive"* / *"event-regime-adaptive"* — see [Reviewer 3, Q5](#5-climatic-inductive-bias--terminology).

### 4. Evaluation does not fully establish the mechanism

> The evaluation does not fully establish the proposed mechanism. Parameter-matched baselines, alternative decompositions, and comparisons between shared and subband-specific temporal modeling are missing. Part of the improvement may also come from the FACL loss rather than the architecture. Moreover, the model does not improve MSE consistently, despite the motivation involving intensity errors at longer lead times.

**A.** We have parameter-matched baselines: AlphaPre, DiffCast, NowcastNet, EarthFarseer.

> ▶️ **RUN:** Collect the parameter counts of all baseline models (AlphaPre, DiffCast, NowcastNet, EarthFarseer, …) and report them.

For other decompositions — Fourier — AlphaPre is the best example, and it makes use of the decomposition in the best possible way. Other transformations could be tried, but doing so would risk losing the motive of why we used wavelets. The model is itself structured to make the best use of the wavelet transform, just as AlphaPre made the best use of the Fourier decomposition. Other transformations similar to wavelet can be explored, which can be posed as a future direction.

> ▶️ **RUN:** Shared-wavelet-parameter experiment — use a single shared wavelet parameter and show why separate is better, proving that separate modeling is actually required.

We also accept that part of the improvement can come from the FACL loss, hence we also showed results with MSE. But we consider that MSE is not the best loss for precipitation nowcasting due to blurring issues (already highlighted by many papers). FACL loss can also serve as motivation to improve precipitation nowcasting. Model development also depends on how best one can make use of the loss being used; in our case the model is orchestrated to make the best use of the FACL loss, resulting in the addition of the Spectral Refinement Spatio-Temporal (SRST) Block rather than some other commonly used spatio-temporal-prediction-specific blocks.

> ▶️ **RUN:** MSE vs CSI — use the CSI–MSE plot.

---

## Potentially decision-changing response

### 1. Controlled comparisons + parameter-matched ablations

> The recommendation could improve with controlled comparisons against recent wavelet and multi-scale nowcasting methods, together with parameter-matched ablations isolating wavelet decomposition, the Gabor stream, SRST, and FACL.

**A.**

> ▶️ **RUN:** Wavelet-based nowcasting model — WADEPre (https://github.com/sonderlau/WADEPre/tree/main). AlphaPre is already included. Others: unknown / TBD.

Stepwise (parameter-matched) ablation showing incremental improvements:

| Model       | Wavelet | Gabor | SRST | FACL |
|-------------|:-------:|:-----:|:----:|:----:|
| Baseline    | ✗       | ✗     | ✗    | ✗    |
| + Wavelet   | ✓       | ✗     | ✗    | ✗    |
| + Gabor     | ✓       | ✓     | ✗    | ✗    |
| + SRST      | ✓       | ✓     | ✓    | ✗    |
| Full Model  | ✓       | ✓     | ✓    | ✓    |

> ▶️ **RUN:** Produce the stepwise ablation numbers for the table above.

### 2. Latent subbands ↔ interpretable radar structures

> Stronger evidence connecting latent subbands to interpretable radar structures would also be important.

**A.**
> ▶️ **CHECK:** See if there is any proof; check the Claude response for this question.

### 3. Why threshold-based metrics improve while MSE does not

> The authors should clarify why threshold-based metrics improve while MSE does not.

**A.** Show the CSI–MSE plot and their correlation — how MSE is lower in the models with high CSI — and highlight the point the creators of Pangu-Weather raised.

> ▶️ **RUN:** Produce the CSI–MSE correlation plot; reference the Pangu-Weather argument on MSE.

### 4. Comparisons with Fourier / learned multi-resolution / shared temporal processing

> Comparisons with Fourier decomposition, learned multi-resolution features, or shared temporal processing would further strengthen the paper.

**A.** For Fourier, see the answer above (AlphaPre is the best example and makes use of the decomposition in the best possible way; other transformations could be tried but would risk losing the motive of why we used wavelets; the model is structured to make the best use of the wavelet transform, just as AlphaPre made the best use of the Fourier decomposition).

Learned multi-resolution features (e.g., U-Net, FPN, multi-scale CNN) — not certain on this, but I think we should study models with implicit multi-resolution capabilities and compare with them as well.

---

# Reviewer 1

### 1. Wavelet transforms are common; add discussion and comparisons

> Utilizing wavelet transforms for multi-scale separation is relatively common in the fields of meteorological forecasting and image processing (e.g., Reference [1]). The authors should supplement the paper with relevant discussions and comparative experiments regarding these prior works.

**A.**
> ▶️ **RUN:** WADEPre.

Respectfully ask the reviewer whether there are any further baselines that utilize wavelet transforms for precipitation nowcasting, which we can also run.

### 2. SimVP already captures multi-scale structure implicitly

> The claim (Line 54) that existing latent-space methods lack physical scale structures is inaccurate. In reality, methods such as SimVP implicitly capture and utilize multi-scale structural information through their downsampling and upsampling network architectures. The authors need to clarify the fundamental differences between their approach and these existing methods.

> **Note:** the claim at Line 54 relates to claim 3; see how it can be linked.

**A.** Better to answer in terms of how we exploit data characteristics that models like SimVP could not. In the paper we wrote *"First, most approaches do not explicitly exploit the multiscale structure of precipitation"* — we refer here to **explicit** decomposition, not the implicit multiscale decomposition done by these models. We also give a separate explanation of exceptions like AlphaPre (lines 48–49).

### 3. SRST drives the gains, not the Gabor core

> The ablation study reveals that removing the SRST Block results in the most significant performance drop. However, the SRST Block is essentially a combination of AFNO and depthwise convolution, which does not constitute a substantial innovation. In contrast, removing the Gabor stream — the paper's primary claimed innovation — leads to a relatively minor performance decrease. This indicates that the model's main performance gains are actually driven by the SRST Block, raising reasonable doubts about the actual effectiveness of the proposed frequency-adaptive core module.

**A.** Answered above ([AC Weakness 1](#1-limited-conceptual-novelty) and [AC Weakness 2](#2-ablation-weakens-the-central-claim-srst-vs-gabor)):

Main contribution: the Gabor activations are adaptable to a particular environmental location. A separate FAT Block for each separately wavelet-decomposed band, with each Gabor adapting to those environmental characteristics, is our novelty. As we see in the sinusoidal-frequency plot, running different sinusoidal patterns for the low and high wavelet frequencies settles at one set of sinusoidal behavior for a given frequency and not another — meaning that modeling a particular wavelet frequency with this particular sinusoidal frequency gives optimal outputs. This leads us to conclude that sinusoids are accurately able to model different wavelet frequencies, leading to SOTA results, and implying that wavelet decomposition and separate Gabor blocks for those wavelets hold significance. One could argue there is not much score difference; this is countered by the fact that the FAT Block has very few parameters compared to the refinement block, so this is expected — hence the limitations section (lines 462–467): *"The learnable parameters of the FAT blocks constitute approximately minor percentage of the total model parameter count. Consequently, the per-dataset ablation score deltas attributable to the adaptive component alone are modest."* and future work (lines 319–320): *"scaling the adaptive part to a larger fraction of model capacity to assess its full potential."* We accept this, but the framework presents a future direction for an adaptive model with SOTA results, acting as motivation for how future precipitation-nowcasting models can adopt a better adaptive approach.

---

# Reviewer 2

### 1. The research gap in lines 45–55 is not a research gap

> "The general problem is clear but the research gap in lines 45-55 is not a research gap."

**A.** The research gap is not a very important section of the paper. The reviewer says the shortcomings are essentially abstract. In support:

- **a.** The second-best model, AlphaPre, has blurring issues, as can be observed in the qualitative plots; also, high intensity could not be captured at higher lead times, as highlighted in the CSI–lead-time plots (Figure 3: Lead-time performance of DAWN-Cast vs baselines on the CIKM dataset) — an issue seen in most convolution-based models.
- **b.** Diffusion-based models like DiffCast do not capture high-intensity positions accurately, indicated by high MSE values; the CSI can also be lower because FN was high.

### B. Describe what other methods are missing

> "This paragraph should describe what other methods are missing."

**A.** This can be highlighted from the qualitative and quantitative plots.

### C. Physical inconsistency vs technical framing; MSE should improve but doesn't

> "For example, if a physical inconsistency would be visible from the SOTA but instead it only talks about technical things." … "…but this sounds like the MSE should be improved which then does not happen."

**A.** Our model's advantage is correct intensity and structure evolution. The model's ability to correctly predict intensity over time is highlighted in the qualitative scores and the additional qualitative plots in the appendix. Even training with MSE gives comparatively better results — e.g., on the SEVIR and Shanghai datasets the improvement is considerable compared to the SOTA model. The slight underperformance on other datasets could be due to insufficient tuning, as the best parameter configuration also depends on the loss; we will include the improved scores in the revised paper if possible. But the high MSE is essentially due to the problem highlighted by Pangu-Weather (as stated above), which is the reason for the high MSE.

### D. Why care about "not explicitly exploiting the multiscale structure"?

> "I am not sure why I should be interested in 'not explicitly exploiting the multiscale structure'."

**A.** Because other models are only able to do so **implicitly**.

### E. Reads more like a report than a Problem → Solution

> "Therefore, this paper feels more like a report on something that was done more than a Problem → Solution structure."

**A.** Yes — this is what we tried to do by explicitly modeling wavelet multiscale behavior using a separate FAT Block and bringing adaptive behavior into the picture as motivation.

### F. The paper fails to put the work into context

> "Further, the paper fails to put the work into context."

**A.** The motivation of the paper was: how can we disintegrate radar images in a meaningful way, and how can a model be designed to use this important disintegrated information for a better purpose? Hence we arrived at the idea of an adaptive inductive bias (and how this inductive bias is introduced is explained above).

> **Note:** apologize for the small formatting mistake in the MSE field of one table.

---

# Reviewer 3

### 1. Main conceptual novelty vs cascade/spectral/multi-resolution priors

> Multi-scale and cascade-based methods are widely studied in precipitation nowcasting. What is the main conceptual novelty of DAWN-Cast compared to prior cascade, spectral, or multi-resolution approaches? Is the key contribution the wavelet decomposition, the latent-space formulation, the FAT block, or their integration?

**A.** What is new is how the sinusoidal behavior inside the FAT Block is able to model the evolution of large-scale convective bulk and small-scale turbulent structures. So the real novelty is in the **Wavelet Guided Temporal Modelling (WGTM) Block**, which includes the wavelet decomposition and Gabor layers taking separate advantage of those separate high- and low-frequency wavelet structures. The motivation of the paper was to present an adaptive approach, but due to the limited parameter count (as stated in the limitations: *"The learnable parameters of the FAT blocks constitute approximately minor percentage of the total model parameter count. Consequently, the per-dataset ablation score deltas attributable to the adaptive component alone are modest."*), and the reason for including the other parts is given in the paragraph above. The Gabor in the FAT Block holds significance (indicated by the sinusoidal-frequency plot / adaptive-Gabor tables).

### 2. Stronger evidence that latent subbands correspond to physical radar structures

> Since wavelet decomposition is applied in latent space, can the authors provide stronger evidence that low-/high-frequency subbands correspond to physically meaningful precipitation structures in the original radar field?

**A.** Latent → original-dataset physical significance, and wavelet application, as prompted by the Claude response.
> ▶️ **CHECK:** find/confirm the latent-to-original physical-significance proof (see [AC decision-changing item 2](#2-latent-subbands--interpretable-radar-structures)).

### 3. Autocorrelation / spectral analyses: original fields, latent, or both?

> Are the autocorrelation and spectral analyses performed on the original radar fields, latent representations, or both? If primarily on original fields, how is the physical interpretation transferred to the latent-space decomposition?

**A.** Autocorrelation was performed in pixel space, just to prove and establish the idea of wavelets. Although the autocorrelation values are from pixel space, we relate the latent to pixel space.
> ▶️ **RUN / CHECK:** find a proof relating latent space to pixel space; and compute the autocorrelation in latent space for all four datasets and report the values.

### 4. Scale-specific temporal modeling vs increased capacity

> To what extent do improvements come from true scale-specific temporal modeling versus increased model capacity? A parameter-matched shared-temporal baseline would help clarify this.

**A.** Ablation + parameters (see the parameter-matched shared-temporal experiment above).
> ▶️ **RUN:** parameter-matched shared-temporal baseline (same as the shared-wavelet-parameter experiment in [AC Weakness 4](#4-evaluation-does-not-fully-establish-the-mechanism)).

### 5. "Climatic" inductive bias — terminology

> The Gabor stream is described as introducing a "climatic" inductive bias. What aspect of the model makes it climate-aware, given that no explicit climate or environmental conditioning appears to be used? Would "frequency-adaptive" or "event-regime-adaptive" be a more precise description?

**A.** The climatic-inductive-bias answer is given above (to be framed). Yes, *frequency-adaptive* and *event-regime-adaptive* are obviously better technical terms, but our main idea was how the Gabor can effectively adapt to the convective-bulk temporal evolution and the turbulence evolution over time.

The sinusoidal behavior: how the different sinusoidal behaviors settle (per the sinusoidal-frequency plot).

### 6. Comparison against alternative multi-scale strategies under comparable budgets

> How does the proposed wavelet-based decomposition compare against alternative multi-scale strategies (e.g., Fourier-based decomposition, cascade models, Laplacian pyramids, or learned multi-resolution feature hierarchies) under comparable computational budgets?

**A.**
> ▶️ **RUN the following comparisons:**
>
> **Fourier:** AlphaPre, FNO, AFNO
>
> **Wavelet:**
> - **WADEPre (2026)** — a wavelet-based decomposition model for extreme precipitation that moves modeling into the wavelet domain, using a dual-branch architecture with an Approximation Network for low-frequency advection and a spatially localized Detail Network for high-frequency stochastic convection. Structurally almost identical to our LL/HF split. (ResearchGate)
> - **WaveC2R (AAAI 2026)** — wavelet-driven coarse-to-refined hierarchical learning, radar-adjacent.
>
> **Cascade models:** CasCast; LDCast (if possible).
>
> **Learned multi-resolution hierarchies:** EarthFormer.

---

# Action Items

Consolidated checklist of everything to **run / check** (see inline `▶️` callouts for context):

- [ ] **Parameter counts** for all baselines (AlphaPre, DiffCast, NowcastNet, EarthFarseer, …) — report table. *(AC W4)*
- [ ] **Shared-wavelet-parameter experiment** — show separate > shared (also the parameter-matched shared-temporal baseline). *(AC W4; Rev3 Q4)*
- [ ] **Stepwise ablation table** — Baseline → +Wavelet → +Gabor → +SRST → Full. *(AC decision-changing 1)*
- [ ] **CSI–MSE plot** + correlation; reference Pangu-Weather argument. *(AC W4; AC decision-changing 3)*
- [ ] **WADEPre** — wavelet-based nowcasting baseline (repo: sonderlau/WADEPre). *(AC decision-changing 1; Rev1 Q1)*
- [ ] **Alternative multi-scale comparisons** — Fourier (FNO, AFNO), Wavelet (WaveC2R), Cascade (CasCast, LDCast), learned multi-res (EarthFormer). *(Rev3 Q6)*
- [ ] **Latent → pixel/physical correspondence proof** — check arXiv:2512.05198 + Claude response. *(AC W3; AC decision-changing 2; Rev3 Q2)*
- [ ] **Latent-space autocorrelation** for all four datasets — report values. *(Rev3 Q3)*
- [ ] Ask **Reviewer 1** for any additional wavelet-based nowcasting baselines to run. *(Rev1 Q1)*
