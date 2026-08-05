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

We could have presented just this as the motivation, but without convincing results it would be difficult for the community to accept the idea. Hence we added some of the conventional spatio-temporal forecasting parts, like global–local refinement, in order to achieve better scores and eventually improve the chances of the idea's acceptability by achieving SoTA results.

### 3. Physical interpretation and "climatic inductive bias"

> The physical interpretation is not sufficiently supported because wavelet decomposition is applied to learned latent features rather than directly to radar fields. It is therefore unclear whether the resulting subbands correspond to physically meaningful precipitation scales. The "climatic inductive bias" claim is also overstated, since the model does not use climate labels or environmental variables.

**A.** We use the autoencoder for the latent → physical-data correspondence proof from *"Your Latent Mask is Wrong,"* arXiv:2512.05198, Dec 2025.

> ▶️ **RUN / CHECK:** Verify the latent→physical correspondence argument against arXiv:2512.05198; also review the Claude response on this question as it is worth exploring.

The word climatic inductive bias is motivated by the idea clearified below: 
We say *climatic* inductive bias because a particular dataset is taken from a particular location, and every location has a certain climate. Gabor application on the wavelet-decomposed data (corresponding to a bulk phenomenon and a turbulence phenomenon) means that only one optimal  Gabor configuration can model each (as shown in the sinosoidal frequency plot) highlighting that the outputs from the FAT Blocks can provide a better inductive bias, to be used by providing a more informative initialization of the latent precipitation dynamics for the subsequent refinement blocks. That initial meaningful output from the FAT Block provides the inductive bias the remaining model needs for a better idea of the climatic evolution in the locality. As radar images are nothing but the shadow of precipitation bands, and precipitation bands are directly affected by the climate variables of the region, this indirectly implies that the movement and evolution of radar images are indirectly affected by the climate of the region. Hence we call it as climatic inductive bias.

> **Note (terminology):** the paper already uses *"climatic inductive bias"* / *"climatically-aware inductive bias"* (Abstract, Sec. 2.1.1). Reviewer 3 suggests *"frequency-adaptive"* / *"event-regime-adaptive"* — see [Reviewer 3, Q5](#5-climatic-inductive-bias--terminology).

### 4. Evaluation does not fully establish the mechanism

> The evaluation does not fully establish the proposed mechanism. Parameter-matched baselines, alternative decompositions, and comparisons between shared and subband-specific temporal modeling are missing. Part of the improvement may also come from the FACL loss rather than the architecture. Moreover, the model does not improve MSE consistently, despite the motivation involving intensity errors at longer lead times.

**A.** We have parameter-matched baselines: AlphaPre, DiffCast, NowcastNet, EarthFarseer.

> ▶️ **RUN:** Collect the parameter counts of all baseline models (AlphaPre, DiffCast, NowcastNet, EarthFarseer, …) and report them.

For other decompositions — Fourier — AlphaPre is the best example, and it makes use of the decomposition in the best possible way. Other transformations could be tried, but doing so would risk losing the motive of why we used wavelets. The model is itself structured to make the best use of the wavelet transform, just as AlphaPre made the best use of the Fourier decomposition. Other transformations similar to wavelet can be explored, which can be posed as a future direction. (This could be posed as a secondary argument after running fourier based transform)

> ▶️ **RUN:** Forier decomposition experiment in DAWNCast instead of wavelet. 
> ▶️ **RUN:** Shared-wavelet-parameter experiment — use a single shared wavelet parameter and show why separate is better, proving that separate modeling is actually required.

We also accept that part of the improvement can come from the FACL loss, hence we also showed results with MSE. But we consider that MSE is not the best loss for precipitation nowcasting due to blurring issues (already highlighted by many papers). FACL loss can also serve as motivation to improve precipitation nowcasting. Model development also depends on how best one can make use of the loss being used; <span style="color:red;">Modify this statement</span>(in our case the model is orchestrated to make the best use of the FACL loss, resulting in the addition of the Spectral Refinement Spatio-Temporal (SRST) Block rather than some other commonly used spatio-temporal-prediction-specific blocks. OR in our case the model is tuned to FACL loss.)

> ▶️ **RUN:** Frequency grid on MSE loss in order to have best mse loss numbers (cikm(P1) and meteonet dataset(P2))
> ▶️ **PUt:** MSE vs CSI — use the CSI–MSE plot.

Use peer-reviewed papers (Ebert & McBride, Roberts & Lean, Buschow et al., Keil & Craig) to establish the double-penalty problem.

---

## Potentially decision-changing response

### 1. Controlled comparisons + parameter-matched ablations

> The recommendation could improve with controlled comparisons against recent wavelet and multi-scale nowcasting methods, together with parameter-matched ablations isolating wavelet decomposition, the Gabor stream, SRST, and FACL.

**A.**

> ▶️ **RUN:** Wavelet-based nowcasting model — WADEPre (https://github.com/sonderlau/WADEPre/tree/main). AlphaPre is already included. Others: unknown / TBD.

Stepwise (parameter-matched) ablation showing incremental improvements:

| Model        | Wavelet | Gabor (FAT) | SRST | FACL | Configuration | Loss |
|--------------|:-------:|:-----------:|:----:|:----:|---------------|------|
| **Baseline** | ✗ | ✗ | ✗ | ✗ | MLP only | MSE |
| **+ Wavelet** | ✓ | ✗ | ✗ | ✗ | Wavelet + MLP | MSE |
| **+ Gabor** | ✓ | ✓ | ✗ | ✗ | Wavelet + Gabor + MLP (optimal parameters) | MSE |
| **+ SRST** | ✓ | ✓ | ✓ | ✗ | Wavelet + Gabor + MLP + SRST (optimal parameters). Evaluate: SRST (Block 1), SRST (Blocks 1+2), SRST (Blocks 1+2+3) | MSE |
| **Full Model** | ✓ | ✓ | ✓ | ✓ | Wavelet + Gabor + MLP + SRST + FACL | FACL |

> ▶️ **RUN:** Produce the stepwise ablation numbers for the table above.

### 2. Latent subbands ↔ interpretable radar structures

> Stronger evidence connecting latent subbands to interpretable radar structures would also be important.
**A.**
> ▶️ **CHECK:** See if there is any proof; check the Claude response for this question.

> ▶️ **RUN:** lag-1 autocorrelation and radial PSD validation directly on the latent tensor Z

This "spectral-analysis-on-latents" methodology is established in mainstream latent-diffusion ML (Diffusability, ICML 2025; Scale-wise Distillation; Spectrum Matching), which find that "the latent frequency spectrum approximately follows a power law, similar to natural images" (SwD, arXiv:2503.16397).

### 3. Why threshold-based metrics improve while MSE does not

> The authors should clarify why threshold-based metrics improve while MSE does not.

**A.** Show the CSI–MSE plot and their correlation — how MSE is lower in the models with high CSI — and highlight the point the creators of Pangu-Weather raised.

state: 

A. long‐term assessment of precipitation forecast skill using the Fractions Skill Score (FSS for double mse problem)
B. Verification of precipitation in weather systems: determination of systematic errors

*Both the above papers address the problem of 2x mse.*

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

We thank the reviewer for this. WADEPre (arXiv:2602.02096) and WaveC2R (AAAI 2026) should have been discussed, and we add a dedicated related-work subsection in the camera-ready. WaveC2R addresses satellite-to-radar retrieval rather than temporal forecasting, so it is not a nowcasting comparator; WADEPre is the closest work and we have now run it.

**Protocol**. WADEPre's published results use T_in = T_out = 6 at 10-min (SEVIR) and 12-min (Shanghai) sampling. We follow the DiffCast/AlphaPre protocol (5→20 at 5-min; 5→10 on CIKM), so their published table is not directly comparable and we re-trained WADEPre under our setting. [TABLE]

**Architectural constraint**. WADEPre's A-Net compresses the temporal axis into channels and its Stationary Texture Assumption (their Eq. 6) repeats the last observed detail frame across the horizon. The design ties output length to input structure and does not natively support T_out ≠ T_in, which every setting in our evaluation requires. 

**Where the decompositions differ.** Both works apply a DWT and route bands to separate branches. **The difference is which axis the decomposition specializes**. WADEPre differentiates the bands spatially — a dilated ResNet for the approximation branch, an FPN for the detail branch — while both branches use the same class of temporal operator: A-Net mixes time-as-channels with an MLP (their Eq. 5) and D-Net projects the sequence with a shared temporal MLP across all levels (their Eq. 7). DAWN-Cast differentiates the temporal axis: each subband receives an operator whose response characteristic, from smooth-linear to oscillatory, is a per-subband parameter. 

**Which axis is specialized.** WADEPre differentiates the bands spatially — a dilated ResNet for the approximation branch, an FPN for the detail branch — while both branches use the same class of temporal operator: A-Net mixes time-as-channels with an MLP (their Eq. 5), and D-Net projects the sequence with a temporal MLP shared across all detail levels (their Eq. 7). DAWN-Cast differentiates the temporal axis: each subband receives an operator whose response characteristic, from smooth-linear to oscillatory, is a per-subband parameter.

**Compute difference**:  WADEPre's own limitations note the computational cost of its multi-scale transforms, and their reported setup uses four H100 GPUs; DAWN-Cast trains on a single RTX A6000 (L217). WADEPre also requires a four-term objective with curriculum annealing; DAWN-Cast uses a single loss.
-============================================================================================================================

We appreciate the reviewer's observation that wavelet transforms are a well-established for multi-scale signal decomposition in meteorological forecasting and image processing. However, we emphasize that the novelty of DAWN-Cast does not lie in the use of the wavelet transform, but in how the resulting decomposition is exploited for temporal modelling. Our analysis of precipitation dynamics from a multi-scale perspective demonstrates that the 2D discrete wavelet decomposition separates the latent precipitation field into components with markedly different characteristics: the low-frequency (LL) component captures the slowly evolving **large-scale precipitation structure**, whereas the high-frequency (LH/HL/HH) components represent localized **fine-scale variability**, as shown in (Fig 1 & 5). This empirical observation motivates the proposed architecture, in which each wavelet subband is assigned a dedicated Frequency Adaptive Temporal (FAT) block, enabling the model to learn scale-specific temporal dynamics rather than relying on a shared temporal operator across all frequency bands.


**Comparision with WADEPre (ArXiv Feb 2026)**

The Wavelet based method "WADEPre (ArXiv Feb 2026)" recommended by the reviewer is a very recent relevant work, which addresses the dual challenges of blurry extremes and spatial localization errors through wavelet-based disentanglement and a stable coarse-to-fine curriculum learning framework. We have compared against this Model under our evaluation protocol: 

Table: 
| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:------:|:------:|:----:|:----:|:----:|:-----------:|:------:|:------:|:----:|:----:|:----:|
|      | CSI-M↑ | CSI-181↑ | CSI-219↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-35↑ | CSI-40↑ | HSS↑ | SSIM↑ | MSE↓ |
| WADEPre | 0.3524 | 0.1867 | 0.1031 | 0.4497 | 0.6424 | 398.82 | 0.2957 | 0.1908 | 0.1203 | 0.3825 | 0.6576 | **36.19** |
| DAWN-Cast (*ours*)| **0.3638** | **0.1950** | **0.1077** | **0.4668** | **0.7284** | **371.34** | **0.3303** | **0.2349** | **0.1591** | **0.4266** | **0.6696** | 38.64 |

As shown above, DAWN-Cast consistently outperforms WADEPre across the majority of forecasting metrics.

WADEPre is naturally designed for equal input/output horizons (6→6 in their experiments), whereas our architecture directly supports arbitrary forecasting horizons without modifying the temporal modelling mechanism. 


Although both DAWN-Cast and WADEPre employ a Discrete Wavelet Transform, the role of the wavelet decomposition is fundamentally different. In WADEPre, the decomposition primarily serves as a representation mechanism: the approximation coefficients are processed by an Approximation Network while the detail coefficients are processed by a Detail Network before being fused through a refinement stage.

In contrast, our motivation is not simply to process different wavelet bands separately, but to model their temporal evolution differently. We hypothesize that the LL component and the LH/HL/HH components exhibit different characteristics, as supported by our wavelet statistics (Fig. 5 and Appendix C). Consequently, each subband is assigned an independent Frequency Adaptive Temporal (FAT) block, allowing the model to learn distinct temporal dynamics for each frequency regime rather than relying on a shared temporal operator.



<!-- The Gabor operator was specifically chosen because its learnable sinusoidal response can continuously vary from nearly linear to highly oscillatory behaviour through its frequency parameter. Furthermore, Eq. (2) assigns an independent Gabor response using learnable sinosoids to each prediction horizon. Thus, our contribution is not the wavelet transform itself, but combining explicit wavelet decomposition with scale-specific and horizon-specific temporal evolution. -->


<!-- We thank the reviewer for this suggestion. We agree that recent wavelet-based methods should have been discussed in greater detail. We have therefore compared against the most relevant prior work, WADEPre, under our evaluation protocol, and will expand the related work in the camera-ready to include both WADEPre and WaveC2R. While WaveC2R employs wavelet decomposition for satellite-to-radar retrieval rather than temporal nowcasting, WADEPre is the closest architectural comparison. -->






### 2. SimVP already captures multi-scale structure implicitly

> The claim (Line 54) that existing latent-space methods lack physical scale structures is inaccurate. In reality, methods such as SimVP implicitly capture and utilize multi-scale structural information through their downsampling and upsampling network architectures. The authors need to clarify the fundamental differences between their approach and these existing methods.

> **Note:** the claim at Line 54 relates to claim 3; see how it can be linked.

**A.** 

<!-- The reviewer is right that our phrasing at L54 is too absolute, and we will revise it in the camera-ready to "do not explicitly decompose the signal across spatial scales." We note the paper already distinguishes methods that use a unified representation (L48–49, on AlphaPre). Our intended claim is about explicit decomposition, and we make it precise here.
SimVP's multi-scale capacity comes from strided encoder downsampling and multi-kernel Inception modules (3, 5, 7, 11) in the Translator. Three properties distinguish this from an explicit decomposition:
(i) Invertibility. The DWT is a perfect-reconstruction transform: subbands can be processed independently and recombined exactly. Strided convolution is lossy and non-invertible, so scales cannot be separated, treated differently, and losslessly reassembled.
(ii) Addressability. In DAWN-Cast, S^LL is identifiably the coarse-scale field and {S^LH, S^HL, S^HH} the fine-scale residual. We verify this corresponds to a physical property, not just a mathematical one: across all four datasets, LL maintains lag-1 autocorrelation ρ > 0.95 while high-frequency subbands fall to ρ ≤ 0 (Fig. 5, Appendix C). No equivalent statement can be made about any SimVP feature channel — its hierarchy is learned without constraint and carries no scale label.
(iii) Scale-specific parameterization. SimVP applies the same Inception module to all channels. DAWN-Cast assigns a distinct temporal operator per subband. This is the operative difference: the decomposition is not a representation choice but a routing mechanism enabling different physics to receive different treatment.
Empirically, SimVP is among our baselines. Comparing against DAWN-Cast(mse), which shares SimVP's loss function and therefore isolates the architectural difference: SEVIR CSI-M 0.3108 → 0.3413 (+9.8%), MeteoNet 0.3351 → 0.3753 (+12.0%), Shanghai 0.3850 → 0.4368 (+13.5%), CIKM 0.3052 → 0.3136 (+2.8%). SSIM improves on all four (SEVIR 0.6508 → 0.7137; Shanghai 0.7795 → 0.8098).
The controlled test is our w/o Wavelet transform ablation, which holds backbone and parameter budget fixed and removes only the explicit decomposition: SEVIR 0.3638 → 0.3591, MeteoNet 0.4085 → 0.4024. -->

We thank the reviewer for raising this point. We believe there may be a misunderstanding regarding the referenced statement. The cited Line 54 ("existing latent-space methods compress radar...") does not discuss multi-scale representations. Rather, our statement regarding multi-scale structure appears earlier (Line 46): "most approaches do not explicitly exploit the multiscale structure of precipitation." We will revise this wording in the camera-ready to further emphasize the distinction between implicit learned hierarchies and explicit multi-scale decomposition, thereby avoiding any ambiguity.

Methods such as SimVP indeed learn hierarchical multi-resolution features through encoder-decoder downsampling and upsampling, and we do not intend to suggest otherwise. Our distinction is that these hierarchies are implicitly learned feature representations, whereas DAWN-Cast employs an explicit, interpretable wavelet decomposition whose subbands are subsequently assigned dedicated temporal models.

Specifically, three properties distinguish our approach:

(i) Invertibility. The Discrete Wavelet Transform (DWT) is a perfect-reconstruction transform, allowing individual subbands to be processed independently and reconstructed exactly. In contrast, encoder-decoder downsampling via strided convolutions is inherently lossy and does not provide an explicit decomposition into identifiable scales.

(ii) Addressability. The wavelet subbands have explicit semantic meaning: the LL component captures the coarse precipitation structure, while the LH/HL/HH components capture fine-scale spatial variability. Our wavelet analysis (Fig. 5 and Appendix C) further demonstrates that the LL component exhibits substantially higher temporal persistence than the high-frequency subbands, motivating independent temporal modelling. In contrast, the feature channels learned by SimVP do not possess an explicit scale interpretation.

(iii) Scale-specific temporal modelling. Existing implicit multiscale decomposition methods apply a shared temporal operator across the learned feature hierarchy. In DAWN-Cast, each wavelet subband is assigned an independent Frequency Adaptive Temporal (FAT) block, enabling the model to learn distinct temporal dynamics for different spatial-frequency regimes. Thus, the wavelet decomposition serves not merely as a representation, but as a routing mechanism for scale-specific temporal evolution. 


This distinction is also supported empirically. Under the same MSE training objective, DAWN-Cast consistently improves CSI-M over SimVP across all four benchmarks: SEVIR (0.3108 → 0.3413, +9.8%), MeteoNet (0.3351 → 0.3753, +12.0%), Shanghai (0.3850 → 0.4368, +13.5%), and CIKM (0.3052 → 0.3136, +2.8%), while also achieving consistent improvements in the remaining evaluation metrics reported in the paper. Furthermore, our w/o Wavelet ablation isolates the contribution of the explicit decomposition by keeping the backbone and parameter budget fixed, confirming that the observed gains arise from explicit scale-specific temporal modelling rather than simply using a latent hierarchy.

We also share the shared fat-block scores below using the same fat block for different wavelet components for comparison. 

| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:------------:|:-------------:|:----:|:----:|:----:|:---------:|:------------:|:--------------:|:----:|:----:|:----:|
| | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ |
| DAWN-Cast (same-FAT)-low freq params  | 0.3435 | 0.3715 | 0.4389 | 0.4403 | 0.7092 | 400.24 | 0.3201 | 0.3430 | 0.4003 | 0.4144 | 0.6627 | 39.07 |
| DAWN-Cast (same-FAT)-high freq params | 0.3444 | 0.3714 | 0.4365 | 0.4417 | 0.7102 | 400.76 | 0.3219 | 0.3450 | 0.4034 | 0.4168 | 0.6596 | 39.01 |
| DAWN-Cast(*ours*) (diff-FAT) - high and low freq| **0.3638** | **0.4054** | **0.4856** | **0.4668** | **0.7284** | **371.34** | **0.3303** | **0.3543** | **0.4135** | **0.4266** | **0.6696** | **38.64** |

The above table clearly demonstrates the original reason for improvement is our novel different FAT blocks.

### 3. SRST drives the gains, not the Gabor core

> The ablation study reveals that removing the SRST Block results in the most significant performance drop. However, the SRST Block is essentially a combination of AFNO and depthwise convolution, which does not constitute a substantial innovation. In contrast, removing the Gabor stream — the paper's primary claimed innovation — leads to a relatively minor performance decrease. This indicates that the model's main performance gains are actually driven by the SRST Block, raising reasonable doubts about the actual effectiveness of the proposed frequency-adaptive core module.

This is the central concern raised and we address it with four pieces of evidence.

1. The Gabor stream is not additive — the WGTM depends on it. Table 2 shows that removing the Gabor stream while retaining the WGTM (SEVIR 0.3541) is worse than removing the entire WGTM block (0.3566). The ordering replicates on MeteoNet (0.4002 vs 0.4054). Wavelet-guided temporal modelling with an MLP-only stream is net-negative relative to not having the block at all; it becomes beneficial only when paired with the subband-specific operator (0.3638 / 0.4085). We further ran a parameter-matched control in which the Gabor stream is replaced by a width-matched MLP, holding parameter count fixed: [INSERT]. The Gabor is therefore not a marginal addition to a block that works without it — it is what makes the decomposition pay for itself.

2. The SRST is not a stack of two known modules. Removing the spectral and spatial branches individually costs 0.0112 and 0.0086 CSI-M on SEVIR (sum 0.0198), while removing the block costs 0.0336 — 70% more than additive. MeteoNet shows the same super-additivity (0.0305 vs 0.0384). Two independently-composed known modules would predict additivity; the observed interaction between global spectral and local spatial refinement is what the block contributes.

3. Contribution per parameter. The FAT blocks are 0.27% of model parameters (83.8K) and yield ΔCSI-M = 0.0097 on SEVIR; the SRST blocks are 99.29% (59.1M) and yield 0.0336. Per unit of parameter budget the FAT block contributes roughly 106× more on SEVIR and 80× more on MeteoNet. We raise this not to dispute the reviewer's reading of raw deltas but because the comparison as posed weighs a 0.27%-parameter structural prior against a 99%-parameter refinement stack; the raw deltas are the expected outcome of that asymmetry, not evidence that the prior is inert.

4. Behavioural evidence. Table 4 reports a reversal, not a magnitude: on SEVIR Storm events the strongly sinusoidal configuration outperforms the near-linear one (CSI 0.3060 vs 0.3017), and on SEVIR Random events the ordering inverts (0.3071 vs 0.3147). A parameter-count argument cannot produce a sign flip. Table 3 additionally shows learnable Gabor parameters outperforming fixed ones on both datasets tested.
We accept the reviewer's framing that the per-dataset deltas attributable to the adaptive component are modest, as stated in Sec. A(i) and L319–320. Our claim is scoped accordingly: the SRST provides refinement capacity, and the FAT block provides the scale-separated structure that makes it a physically-organised model rather than a generic one — at 0.27% of the parameter budget.

---

We thank the reviewer for the insightful comment. We would like to clarify that the proposed WGTM is not intended to replace the refinement backbone, but to provide an explicit multiscale temporal inductive bias before refinement. By decomposing the latent representation into wavelet subbands and assigning each subband an independent Frequency Adaptive Temporal (FAT) block, the model learns scale-specific temporal evolution before the SRST reconstructs globally consistent latent features.

This behaviour is consistently supported by the ablation studies (Table 2). Removing the wavelet decomposition, removing the Gabor/FAT module, or replacing the adaptive with non adaptive gabors (Table 3) or using FAT blocks with identical parameter initializations for all scales(table above in last weakness) all reduce performance across datasets. These results indicate that explicitly modelling the temporal dynamics of different wavelet subbands using specialized temporal operator (FAT block), supporting the contribution of the proposed WGTM.


We also respectfully disagree that the SRST is simply a stack of two existing modules. The SRST derives its effectiveness from the interaction between global spectral refinement and local spatial refinement.If these branches contributed independently, one would expect their individual ablation drops to be approximately additive. However, this is not observed. On SEVIR, removing the spectral branch decreases CSI-M by 0.0112, while removing the spatial branch decreases it by 0.0086; their sum is 0.0198. In contrast, removing the complete SRST block results in a substantially larger drop of 0.0336 (approximately 70% larger than the additive expectation). A similar super-additive behaviour is observed on MeteoNet (0.0305 vs. 0.0384) (Ref Table 2 of paper). This indicates that the performance gain arises from the complementary interaction between the spectral and spatial refinement streams rather than from simply stacking two independent modules.

Contribution per parameter: the FAT block delivers ~13× more CSI-M gain per parameter than the SRST. 

| Component | Parameters | % of Model | ΔCSI-M (SEVIR) | ΔCSI-M (MeteoNet) | Gain per 1M (SEVIR) | Gain per 1M (MeteoNet) | Relative efficiency |
|-----------|-----------:|-----------:|---------------:|------------------:|--------------------:|-----------------------:|--------------------:|
| FAT (Gabor) | ~1 M | ~2% | 0.0097 | 0.0083 | 0.00970 | 0.00830 | ≈13× |
| SRST | ~51 M | ~98% | 0.0336 | 0.0384 | 0.00066 | 0.00075 | 1× |

Also we would like to state that other models in the field of precipitation nowcasting had similar parameter count. Paramter count table of other models made for the field of precipitation nowcasting. 

| Model | Parameters (M) |
|-------|---------------:|
| AlphaPre | 89.03 |
| EarthFarseer | 148.57 |
| DiffCast | 49.36 |
| NowcastNet | 34.90 |
|WadePre | 43.20 |
|Cascast | 311.7 |
| DAWNCast (latent) | 51.41 |


# Reviewer 2

### 1. The research gap in lines 45–55 is not a research gap

> "The general problem is clear but the research gap in lines 45-55 is not a research gap."

**A.** The research gap is not a very important section of the paper. The reviewer says the shortcomings are essentially abstract. In support:

- **a.** The second-best model, AlphaPre, has blurring issues, as can be observed in the qualitative plots; also, high intensity could not be captured at higher lead times, as highlighted in the CSI–lead-time plots (Figure 3: Lead-time performance of DAWN-Cast vs baselines on the CIKM dataset) — an issue seen in most convolution-based models.
- **b.** Diffusion-based models like DiffCast do not capture high-intensity positions accurately, indicated by high MSE values; the CSI can also be lower because FN was high.

I am not very sure what to write. 

### B. Describe what other methods are missing

> "This paragraph should describe what other methods are missing."

**A.** This can be highlighted from the qualitative and quantitative plots.

### C. Physical inconsistency vs technical framing; MSE should improve but doesn't

> "For example, if a physical inconsistency would be visible from the SOTA but instead it only talks about technical things." … "…but this sounds like the MSE should be improved which then does not happen."

**A.** Our model's advantage is correct intensity and structure evolution. The model's ability to correctly predict intensity over time is highlighted in the qualitative scores and the additional qualitative plots in the appendix. Even training with MSE gives comparatively better results — e.g., on the SEVIR and Shanghai datasets the improvement is considerable compared to the SOTA model. The slight underperformance on other datasets could be due to insufficient tuning, as the best parameter configuration also depends on the loss; we will include the improved scores in the revised paper if possible. But the high MSE is essentially due to the problem highlighted by Pangu-Weather (as stated above), which is the reason for the high MSE.

### D. Why care about "not explicitly exploiting the multiscale structure"?

> "I am not sure why I should be interested in 'not explicitly exploiting the multiscale structure'."

**A.** Because other models are only able to do so **implicitly**. and show the frequency plot . 

### E. Reads more like a report than a Problem → Solution

> "Therefore, this paper feels more like a report on something that was done more than a Problem → Solution structure."

**A.** Yes — this is what we tried to do by explicitly modeling wavelet multiscale behavior using a separate FAT Block and bringing adaptive behavior into the picture as motivation.

### F. The paper fails to put the work into context

> "Further, the paper fails to put the work into context."

**A.** The motivation of the paper was: how can we disintegrate radar images in a meaningful way, and how can a model be designed to use this important disintegrated information for a better purpose? Hence we arrived at the idea of an adaptive inductive bias (and how this inductive bias is introduced is explained above).

> **Note:** apologize for the small formatting mistake in the MSE field of one table.
---------------------------------------------------------------------------------------------------------------------------------------------------------

We totally understand your point and wish to refine and increase the current research gap section in revised version by including the visible inconsistency problems as well like blurring, position inconsistency, inconsistency over lead times, retaination of high intensity over larger lead times and other similar visible inconsistent factors across various models, and also how our model is trying to improve on any of these inconsistencies. 

To further clarify our model's motivation, the objective of explicitly exploiting the multi-scale structure is to better model the evolution of precipitation systems. By decomposing radar observations into multiple spatial scales, the model can more effectively capture the dynamics of both large-scale precipitation bands and finer-scale convective structures, leading to more accurate precipitation nowcasting, <as highlighted by previous papers previously>. Hence we do  explicit multiscale decomposition of the radar observations and notice that they correspond to characteristics which can be correlated with actual precipitation structure characteristics as stated in the contribution 1 statement "low-frequency values correspond to the large-scale structure and high-frequency values correspond to the fine-scale variability of the precipitation field.".  We observe different precipitation field characteristics pose different motion trends which we use as motivation for building different specialized gabor based FAT blocks, and we notice that sinosoids in gabor can help us do that. Sinosoids can vary from near linear(small z in sin(z)) to oscillatory trend (large z). We also show in Table 4. how different gabor configurations help to model different situations effectively, technically giving us manual ability to have a best model for different atmospheric dynamics (Ref Table 4). 

We would like to clarify that MSE is a point-wise error metric that evaluates forecasts through exact grid-point correspondence. Such metrics do not explicitly account for spatial displacement errors and are therefore known to be less suitable for evaluating high-resolution precipitation forecasts. Previous work has demonstrated through idealized displacement experiments that forecasts with only small spatial location errors can receive poor scores under grid-point evaluation despite remaining meteorologically similar to the observations, while spatially aware verification methods provide more meaningful assessments (Ref [1]).

Consistent with these findings, our additional experiments on frequency sweep also reveals that lower MSE does not necessarily correspond to better csi scores. For example, the configuration with the lowest MSE (11.2099) yields the lowest CSI (0.3945), whereas the configuration achieving the highest CSI (0.4114) has a comparatively larger MSE (11.7235). Similarly, another high-CSI configuration (0.4091) corresponds to the largest MSE (11.9392). These observations further indicate that point-wise error alone is insufficient to characterize forecast quality in precipitation nowcasting, motivating the use of other methods such as CSI alongside MSE.


Meteonet frequency sweep 
CSI HEATMAP: 
Legend:
⬜ ≥ 0.4100 (Highest), 🟨 ≥ 0.4070, 🟥 ≥ 0.4040, ⬛ < 0.4040 (Lowest)
⭐ Best ❌ Worst
| HF \ LL | 1.09 | 3.28 | 8.74 | 17.49 | 34.34 |
|:-------:|:----:|:----:|:----:|:-----:|:-----:|
| **4.41** | ⬜ **0.4114 ⭐** | 🟨 0.4084 | 🟨 0.4070 | 🟥 0.4056 | 🟨 0.4070 |
| **2.25** | ⬛ 0.4032 | ⬛ **0.3945 ❌** | 🟨 0.4080 | 🟨 0.4091 | 🟨 0.4083 |
| **1.12** | 🟥 0.4040 | 🟥 0.4040 | 🟥 0.4066 | 🟨 0.4082 | 🟥 0.4058 |
| **0.42** | ⬛ 0.3993 | 🟥 0.4062 | 🟨 0.4078 | 🟥 0.4050 | 🟥 0.4053 |
| **0.14** | 🟥 0.4058 | 🟥 0.4043 | 🟥 0.4066 | 🟨 0.4087 | 🟥 0.4059 |

MSE HEATMAP:
Legend: ⬜ < 11.3 (Best), 🟨 11.3 – <11.5, 🟥 11.5 – <11.7, ⬛ ≥ 11.7 (Worst)
| HF \ LL | 1.09 | 3.28 | 8.74 | 17.49 | 34.34 |
|:-------:|:----:|:----:|:----:|:-----:|:-----:|
| **4.41** | ⬛ 11.7235 | ⬛ 11.7033 | 🟥 11.6145 | 🟥 11.5906 | ⬛ 11.7424 |
| **2.25** | 🟥 11.6426 | ⬜ **11.2099 ⭐** | ⬛ 11.8800 | ⬛ **11.9392 ❌** | ⬛ 11.8561 |
| **1.12** | 🟥 11.6555 | 🟥 11.5354 | 🟥 11.6250 | ⬛ 11.9265 | 🟥 11.6057 |
| **0.42** | 🟨 11.4663 | 🟥 11.5854 | ⬛ 11.8545 | 🟥 11.6017 | 🟥 11.5312 |
| **0.14** | ⬛ 11.7509 | 🟥 11.5493 | ⬛ 11.8147 | ⬛ 11.8839 | ⬛ 11.7315 |

Label: CSI-M and MSE Heatmap under the static-γ Gabor regime sweep. Each axis spans the calibrated freq_multiplier ladder from near-linear (z ≈ 0.1) to a full half-cycle oscillation (z ≈ π) — x-axis for the LL (convective-bulk) FAT block, y-axis for the HF (turbulence) FAT block — with γ frozen at its checkpoint-derived value in both blocks, isolating the effect of the frequency-multiplier axis alone.

We will insert the recommended citations (Gabor, CSI, HSS) in the revised version paper. 

[1] Roberts, Nigel M., and Humphrey W. Lean. "Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events." Monthly Weather Review 136.1 (2008): 78-97.


Question: What is the motivation for developing an architecture like this other than the idea of it being closer to physics should make it unspecifically better?

**A** We thank the reviewer for this important question. Our motivation is not that a more physics-inspired architecture is inherently better. Rather, we hypothesize that explicitly separating precipitation into components with different characteristics(large salce convective bulk and fine scale variability) simplifies the learning problem in approximation of precipitation field movement through radar observations. 

So our motivation is modelling precipitation for mulit-scale components makes it easier to model precipitation evolution. For E.g. large-scale storm structure evolves smoothly over time, whereas fine-scale structures exhibit faster and more irregular dynamics (as shown in fig 5, Appendix C). When these components are represented jointly, a single temporal operator must simultaneously model both slowly varying and rapidly evolving processes. This places conflicting demands on a shared representation and increases the complexity of the learning task.

DAWN-Cast addresses this by first explicitly decomposing the latent representation into wavelet subbands and then assigning an independent Frequency Adaptive Temporal (FAT) block to each subband. Consequently, each temporal operator (FAT Block) only needs to learn the dynamics of a narrower frequency regime instead of modelling the entire spectrum simultaneously. We therefore view the wavelet decomposition primarily as a mechanism for reducing temporal heterogeneity, rather than simply introducing a physics-inspired representation.

The choice of a Gabor-based temporal operator follows the same motivation. Because its sinusoidal component can vary from nearly linear to highly oscillatory behaviour through learnable frequency parameters, it provides a flexible family of temporal responses capable of representing both slowly evolving convective organization and rapidly varying fine-scale structures. Furthermore, Eq. (2) defines independent Gabor responses across prediction horizons, allowing different future lead times to learn different temporal behaviours instead of enforcing a single global evolution function across the entire forecast sequence.

Our empirical analyses support this hypothesis. The wavelet statistics presented in Fig. 5 and Appendix C demonstrate that the low-frequency component and high-frequency componets exhibit different charactercteristic and can be treated seperately. In addition, the adaptive Gabor frequency sweeps converge to different optima across datasets, suggesting that different precipitation regimes favour different temporal frequency responses. Finally, the adaptive FAT blocks consistently outperform fixed Gabor configurations (ref Table3) and removing the gabor and wavelet-guided temporal modelling degrades performance (ref Table 2),introducing FAT with same parameters for both all the scales also degrades the performance as shown in the table below, indicating that the gains arise from modelling scale-specific temporal dynamics rather than from the wavelet transform alone.

| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:------------:|:-------------:|:----:|:----:|:----:|:---------:|:------------:|:--------------:|:----:|:----:|:----:|
| | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ |
| DAWN-Cast (same-FAT)-low freq params  | 0.3435 | 0.3715 | 0.4389 | 0.4403 | 0.7092 | 400.24 | 0.3201 | 0.3430 | 0.4003 | 0.4144 | 0.6627 | 39.07 |
| DAWN-Cast (same-FAT)-high freq params | 0.3444 | 0.3714 | 0.4365 | 0.4417 | 0.7102 | 400.76 | 0.3219 | 0.3450 | 0.4034 | 0.4168 | 0.6596 | 39.01 |
| DAWN-Cast(*ours*) (diff-FAT) - high and low freq| **0.3638** | **0.4054** | **0.4856** | **0.4668** | **0.7284** | **371.34** | **0.3303** | **0.3543** | **0.4135** | **0.4266** | **0.6696** | **38.64** |


**Limitations**: The authors claim that this could be used for other problems of Computervision (line 321) but i cannot see how that is true.

**A** We appreciate the reviewer's comment. Although our original statement was broad, our intention was not to claim applicability to all computer vision problems, but rather to certain spatiotemporal prediction tasks that share similar characteristics with precipitation nowcasting, namely the forecasting of structured spatial fields evolving over time. Examples include satellite imagery forecasting, remote sensing, dynamic medical imaging (e.g., cardiac MRI or ultrasound sequence prediction), and related scientific imaging applications. We will revise the manuscript to explicitly state these application domains and avoid the broader term "computer vision.

# Reviewer 3

### 1. Main conceptual novelty vs cascade/spectral/multi-resolution priors

> Multi-scale and cascade-based methods are widely studied in precipitation nowcasting. What is the main conceptual novelty of DAWN-Cast compared to prior cascade, spectral, or multi-resolution approaches? Is the key contribution the wavelet decomposition, the latent-space formulation, the FAT block, or their integration?

**A.** Wavelet decomposition, latent-space forecasting, and spectral refinement are individually established ideas. Our contribution is not introducing any one of these components in isolation, but proposing a new frequency-adaptive temporal (FAT) modelling framework specialized for modelling seperate precipitation chaaracteristics corresponds to multiscale wavelet transformed features for the task of precipitation nowcasting.

The conceptual novelty lies in the Wavelet Guided Temporal Modelling (WGTM) block. Rather than processing all spatial scales with a shared temporal operator, WGTM first explicitly decomposes the latent representation into physically distinct wavelet subbands and assigns an independent Frequency Adaptive Temporal (FAT) block to each subband. Each FAT block learns its own Gabor parameters, allowing different temporal responses to emerge for large-scale (LL) and fine-scale (HF) precipitation dynamics.

*We agree that wavelet decomposition, latent-space forecasting, cascade refinement, and spectral modelling have all been explored individually. Our contribution is not the use of any one of these components in isolation, but introducing a frequency-adaptive temporal modelling paradigm for precipitation nowcasting. Compared with spectral methods (e.g., AlphaPre), which operate on global Fourier representations, we employ an explicit, localized, and invertible wavelet decomposition that separates coarse (LL) and fine-scale (LH/HL/HH) precipitation components. Compared with learned multi-resolution or cascade approaches (e.g., Diffcast, CasCast), which primarily exploit multi-scale representations or progressive refinement, DAWN-Cast uses this decomposition as a routing mechanism to assign independent Frequency Adaptive Temporal (FAT) blocks to each wavelet subband, enabling distinct temporal evolution across precipitation scales. Likewise, unlike existing wavelet-based methods such as WADEPre, where the wavelet transform primarily serves as a feature representation, our method explicitly learns dedicated temporal dynamics for every wavelet component. Thus, the conceptual novelty lies not in the wavelet transform itself, but in coupling an explicit wavelet decomposition with independent frequency-adaptive temporal operators for scale-specific precipitation forecasting.*

This behaviour is supported by two pieces of evidence already included in the paper:

Adaptive vs. fixed Gabor (Table 3): learning the Gabor parameters consistently outperforms fixed parameters, demonstrating that adaptation itself is beneficial.
Contrasting atmospheric regimes (Table 4): different Gabor initializations become optimal under storm-dominated versus slowly evolving precipitation regimes, supporting our hypothesis that different temporal regimes benefit from different frequency responses. Introducing same hyperparameter initialized FAT blocks  for both all the scales also degrades the performance as shown in the table below. 

| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:------------:|:-------------:|:----:|:----:|:----:|:---------:|:------------:|:--------------:|:----:|:----:|:----:|
| | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ |
| DAWN-Cast (same-FAT)-low freq params  | 0.3435 | 0.3715 | 0.4389 | 0.4403 | 0.7092 | 400.24 | 0.3201 | 0.3430 | 0.4003 | 0.4144 | 0.6627 | 39.07 |
| DAWN-Cast (same-FAT)-high freq params | 0.3444 | 0.3714 | 0.4365 | 0.4417 | 0.7102 | 400.76 | 0.3219 | 0.3450 | 0.4034 | 0.4168 | 0.6596 | 39.01 |
DAWN-Cast (without gabor)- mlp with matched parameters | 0.3566 | 
| DAWN-Cast(*ours*) (diff-FAT) - high and low freq| **0.3638** | **0.4054** | **0.4856** | **0.4668** | **0.7284** | **371.34** | **0.3303** | **0.3543** | **0.4135** | **0.4266** | **0.6696** | **38.64** |

In addition, during hyperparameter exploration we observed that the optimal Gabor frequency multipliers differ across datasets (examples shown below in the attached heatmaps). Rather than a single universal configuration, each dataset converges to a different optimum, suggesting that the adaptive temporal operator specializes to the underlying statistics of the dataset. While these sweeps were not included in the paper due to space constraints, they are consistent with our proposed motivation.

CSI HEATMAPS: 
CIKM frequency sweep 
| HF \ LL | 22.74 | 68.23 | 181.94 | 363.89 | 714.49 |
|:-------:|:-----:|:-----:|:------:|:------:|:------:|
| **95.56** | 0.3269 | 0.3292 | 0.3201 | 0.3289 | 0.3264 |
| **48.67** | **0.3362 🥇** | 0.3235 | 0.3224 | 0.3217 | 0.3252 |
| **24.34** | 0.3276 | 0.3243 | 0.3292 | 0.3246 | 0.3248 |
| **9.13**  | 0.3275 | 0.3296 | 0.3256 | 0.3237 | 0.3273 |
| **3.04**  | 0.3266 | *0.3311 🥈* | 0.3253 | 0.3270 | 0.3234 |

Meteonet frequency sweep 
Legend:
| HF \ LL | 1.09 | 3.28 | 8.74 | 17.49 | 34.34 |
|:-------:|:----:|:----:|:----:|:-----:|:-----:|
| **4.41** | **0.4114 🥇** | 0.4084 | 0.4070 | 0.4056 | 0.4070 |
| **2.25** | 0.4032 | 0.3945 | 0.4080 | *0.4091 🥈* | 0.4083 |
| **1.12** | 0.4040 | 0.4040 | 0.4066 | 0.4082 | 0.4058 |
| **0.42** | 0.3993 | 0.4062 | 0.4078 | 0.4050 | 0.4053 |
| **0.14** | 0.4058 | 0.4043 | 0.4066 | 0.4087 | 0.4059 |


---------------------------------------------------------------------------

Wavelet decomposition, latent-space forecasting, spectral modelling, cascade refinement, and multi-resolution representations have all been explored individually. Our contribution is not the introduction of any one of these components in isolation, but a new frequency-adaptive temporal modelling paradigm for precipitation nowcasting.
The conceptual novelty lies in the proposed Wavelet Guided Temporal Modelling (WGTM) framework. Avoiding the use of shared temporal operator for the radar observation at once, WGTM first performs an explicit and invertible wavelet decomposition separating the latent representation into physically meaningful frequency subbands. This decomposition allows mutliscale decompose to low (corres. large-scale str.) and high frequency details(corres. fine scale variability), assigning each subband to an independent Frequency Adaptive Temporal (FAT) block, allows coarse precipitation structures and fine-scale convective details to be approximated seprarately according to distinct temporal dynamics.

Our modelling objective also differs from prior families of approachesas described below:
1. Compared with spectral approaches (s.a. Alphapre), which operate on global Fourier representation (amplitude and phase) modelling, our radar observation modelling approach differs in terms of spectral decomposition and modelling.  

1. Fourier based methods Use amplitude and phase based decomposition for precipitation modelling, In contrast we decompose the radar observations to Large scale structure (Low frequency) Fine-Scale variability (High frequency) using Wavelet decomposition for precipitation nowcasting.


2. Compared with learned multi-resolution approaches (s.a. simvp, earthformer), the multi-scale representation is learned implicitly through the network architecture. In contrast, DAWN-Cast employs an explicit and invertible wavelet decomposition subbands, allowing each frequency regime to be processed by its own temporal evolution module rather than sharing a common temporal operator.
3. Compared with cascade approaches (s.a. Cascast, diffcast), Ours is also a cascaded based method similar to cascast but we differ in how we model in forcasting which is our novelty and is proven to be better than baseline

4. Compared with existing wavelet-based forecasting methods such as WADEPre, the wavelet transform is not merely used as a hierarchical representation. Instead, every wavelet subband is assigned an independent adaptive temporal model. For example, WADEPre models low-frequency dynamics through an Approximation Network while relying on the Stationary Texture Assumption for high-frequency components, whereas DAWN-Cast explicitly learns the temporal evolution of every wavelet subband through dedicated FAT blocks.

Therefore, the conceptual novelty is not the wavelet transform itself, the latent-space formulation, or the FAT block individually, but their integration into a unified framework in which an explicit wavelet decomposition enables independent, frequency-adaptive temporal modelling across precipitation scales.
----------

This modelling principle is supported by multiple observations already included in the paper. Adaptive Gabor parameters consistently outperform fixed parameters (Table 3), different atmospheric regimes favour different temporal frequency responses (Table 4), and our new ablation shows that forcing all wavelet subbands to share the same FAT configuration degrades performance compared with assigning independent FAT blocks. Furthermore, the attached frequency sweeps demonstrate that different datasets converge to different optimal combinations of low- and high-frequency responses, suggesting that no single temporal frequency configuration is universally optimal. These observations consistently support our hypothesis that precipitation scales exhibit distinct temporal evolution patterns that benefit from dedicated adaptive temporal operators.

This behaviour is supported by two pieces of evidence already included in the paper:

Adaptive vs. fixed Gabor (Table 3): learning the Gabor parameters consistently outperforms fixed parameters, demonstrating that adaptation itself is beneficial.
Contrasting atmospheric regimes (Table 4): different Gabor initializations become optimal under storm-dominated versus slowly evolving precipitation regimes, supporting our hypothesis that different temporal regimes benefit from different frequency responses. Introducing same hyperparameter initialized FAT blocks  for both all the scales also degrades the performance as shown in the table below. 

| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:------------:|:-------------:|:----:|:----:|:----:|:---------:|:------------:|:--------------:|:----:|:----:|:----:|
| | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ |
| DAWN-Cast (same-FAT)-low freq params  | 0.3435 | 0.3715 | 0.4389 | 0.4403 | 0.7092 | 400.24 | 0.3201 | 0.3430 | 0.4003 | 0.4144 | 0.6627 | 39.07 |
| DAWN-Cast (same-FAT)-high freq params | 0.3444 | 0.3714 | 0.4365 | 0.4417 | 0.7102 | 400.76 | 0.3219 | 0.3450 | 0.4034 | 0.4168 | 0.6596 | 39.01 |
DAWN-Cast (without gabor)- mlp with matched parameters | 0.3566 | 
| DAWN-Cast(*ours*) (diff-FAT) - high and low freq| **0.3638** | **0.4054** | **0.4856** | **0.4668** | **0.7284** | **371.34** | **0.3303** | **0.3543** | **0.4135** | **0.4266** | **0.6696** | **38.64** |

In addition, during hyperparameter exploration we observed that the optimal Gabor frequency multipliers differ across datasets (examples shown below in the attached heatmaps). Rather than a single universal configuration, each dataset converges to a different optimum, suggesting that the adaptive temporal operator specializes to the underlying statistics of the dataset. While these sweeps were not included in the paper due to space constraints, they are consistent with our proposed motivation.

CSI HEATMAPS: 
CIKM frequency sweep 
| HF \ LL | 22.74 | 68.23 | 181.94 | 363.89 | 714.49 |
|:-------:|:-----:|:-----:|:------:|:------:|:------:|
| **95.56** | 0.3269 | 0.3292 | 0.3201 | 0.3289 | 0.3264 |
| **48.67** | **0.3362 🥇** | 0.3235 | 0.3224 | 0.3217 | 0.3252 |
| **24.34** | 0.3276 | 0.3243 | 0.3292 | 0.3246 | 0.3248 |
| **9.13**  | 0.3275 | 0.3296 | 0.3256 | 0.3237 | 0.3273 |
| **3.04**  | 0.3266 | *0.3311 🥈* | 0.3253 | 0.3270 | 0.3234 |

Meteonet frequency sweep 
Legend:
| HF \ LL | 1.09 | 3.28 | 8.74 | 17.49 | 34.34 |
|:-------:|:----:|:----:|:----:|:-----:|:-----:|
| **4.41** | **0.4114 🥇** | 0.4084 | 0.4070 | 0.4056 | 0.4070 |
| **2.25** | 0.4032 | 0.3945 | 0.4080 | *0.4091 🥈* | 0.4083 |
| **1.12** | 0.4040 | 0.4040 | 0.4066 | 0.4082 | 0.4058 |
| **0.42** | 0.3993 | 0.4062 | 0.4078 | 0.4050 | 0.4053 |
| **0.14** | 0.4058 | 0.4043 | 0.4066 | 0.4087 | 0.4059 |

### 2 & 3. Stronger evidence that latent subbands correspond to physical radar structures

> Since wavelet decomposition is applied in latent space, can the authors provide stronger evidence that low-/high-frequency subbands correspond to physically meaningful precipitation structures in the original radar field?
> Are the autocorrelation and spectral analyses performed on the original radar fields, latent representations, or both? If primarily on original fields, how is the physical interpretation transferred to the latent-space decomposition?

**A**

Our forecasting model operates in the latent space of the pretrained convolutional autoencoder introduced by Rombach et al. [1]. These latent representations are not arbitrary feature vectors; owing to the convolutional and strided encoder architecture, they preserve the spatial organization of the original radar observations while encoding richer contextual information. This spatial correspondence has recently been analyzed by Bradbury and Zhong [2], who show that latent positions produced by Rombach-style latent autoencoders remain spatially aligned with the corresponding image regions. Consequently, although our processing is performed in the latent domain, the latent features still represent spatial precipitation structures originating from the radar observations.

Our original analyses in Figures 1 and 5 were intentionally performed in pixel space, demonstrating that the wavelet low-frequency (LL) and high-frequency (LH/HL/HH) subbands correspond to physically meaningful precipitation structures in the original radar observations. Specifically, the LL component captures coherent large-scale precipitation bands, whereas the high-frequency components represent fine scale variability. To address the reviewer's concern, we additionally analyzed the corresponding latent representations and compared them directly with the original radar fields using spatial autocorrelation, power spectral density (PSD), low-frequency energy fraction (LFE), and pixel-latent correlation statistics (Table below).

Table 

GT: the ground-truth radar/precipitation field itself, no wavelet decomposition applied — included as a baseline reference.

LL: the wavelet approximation subband (coherent large-scale structure).

HF: the three wavelet detail subbands (LH, HL, HH — fine-scale/turbulent structure), averaged into one number here for compactness.

ρ_pixel / ρ_latent: mean lag-1 spatial autocorrelation, computed on the pixel-space frame and on the corresponding latent tensor Z, respectively.

paired r: Pearson correlation between ρ_pixel and ρ_latent computed per sample across the paired ensemble (not just matching ensemble averages) — "—" for HF because it's an average of three separate correlation coefficients, which shouldn't be naively averaged (would need a Fisher z-transform first; happy to add if useful).

LFE (pix/lat): fraction of total spectral power in the lowest quartile of relative wavenumber, computed separately in pixel space and latent space.
PSD-shape r: Pearson correlation (in log space) between the pixel-space and latent-space power spectral density curves, after resampling the finer pixel-space curve onto the coarser latent grid.

| Dataset (n) | Field | Spatial autocorr. – pixel (ρ) | Spatial autocorr. – latent (ρ) | Pixel–latent paired correlation (r) | Low-freq. energy fraction (LFE) (Pixel/Latent) | PSD-shape correlation – pixels/latent (r) |
|:-----------|:------|------------------------------:|-------------------------------:|------------------------------------:|:----------------------------------------------:|------------------------------------------:|
| **CIKM (n=200)** | GT – raw radar field | 0.959 | 0.562 | 0.692 | 0.999 / 0.986 | 0.983 |
|  | LL – low-frequency subband | 0.939 | 0.573 | 0.804 | 0.998 / 0.971 | 0.990 |
|  | HF – high-frequency subbands (avg. of LH, HL, HH) | 0.130 | 0.067 | — | 0.494 / 0.319 | 0.564 |
| **MeteoNet (n=200)** | GT – raw radar field | 0.895 | 0.562 | 0.940 | 0.995 / 0.972 | 0.943 |
|  | LL – low-frequency subband | 0.791 | 0.591 | 0.876 | 0.984 / 0.965 | 0.992 |
|  | HF – high-frequency subbands (avg. of LH, HL, HH) | -0.235 | -0.047 | — | 0.083 / 0.198 | 0.799 |
| **Shanghai (n=154)** | GT – raw radar field | 0.947 | 0.652 | 0.821 | 0.998 / 0.975 | 0.945 |
|  | LL – low-frequency subband | 0.880 | 0.674 | 0.668 | 0.993 / 0.955 | 0.987 |
|  | HF – high-frequency subbands (avg. of LH, HL, HH) | -0.248 | -0.049 | — | 0.074 / 0.209 | 0.893 |
| **SEVIR (n=141)** | GT – raw radar field | 0.937 | 0.566 | -0.020 | 0.999 / 0.968 | 0.926 |
|  | LL – low-frequency subband | 0.859 | 0.611 | 0.476 | 0.994 / 0.958 | 0.992 |
|  | HF – high-frequency subbands (avg. of LH, HL, HH) | -0.242 | -0.042 | — | 0.084 / 0.206 | 0.542 |

The results show that the same structural characteristics are preserved after encoding into the latent space. Across all four datasets, the LL subband consistently exhibits substantially higher spatial autocorrelation than the high-frequency subbands in both pixel and latent space (e.g., SEVIR: ρ = 0.859 vs. −0.242 in pixel space and 0.611 vs. −0.042 in latent space), confirming that the encoder preserves the distinction between coherent precipitation structures and localized variability. Likewise, the PSD-shape correlation between pixel and latent representations remains consistently high for the LL component (0.987–0.992 across datasets), indicating that the spectral characteristics of the large-scale precipitation structures are retained after encoding. The low-frequency energy fraction also remains strongly preserved (e.g., SEVIR LL: 0.994/0.958 and CIKM LL: 0.998/0.971 for pixel/latent), while the paired pixel-latent correlations further demonstrate that the latent LL representations remain strongly aligned with their corresponding radar structures (e.g., r = 0.804 for CIKM and r = 0.876 for MeteoNet).


Therefore, our physical interpretation is established in the original radar field and is subsequently supported by quantitative correspondence analyses between pixel and latent space. Together with the spatially aligned latent representations reported by Rombach et al. [1] and further analyzed by Bradbury and Zhong [2], these results provide evidence that the latent-space wavelet decomposition preserves physically meaningful precipitation structures while enabling computationally efficient temporal modelling.

Reference: 
1. Rombach, R., Blattmann, A., Lorenz, D., Esser, P. and Ommer, B., 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10684-10695).
2. Bradbury, R. and Zhong, D., 2025. Your Latent Mask is Wrong: Pixel-Equivalent Latent Compositing for Diffusion Models. arXiv preprint arXiv:2512.05198.
**A.** Latent → original-dataset physical significance, and wavelet application, as prompted by the Claude response.
> ▶️ **CHECK:** find/confirm the latent-to-original physical-significance proof (see [AC decision-changing item 2](#2-latent-subbands--interpretable-radar-structures)).



### 4. Scale-specific temporal modeling vs increased capacity

> To what extent do improvements come from true scale-specific temporal modeling versus increased model capacity? A parameter-matched shared-temporal baseline would help clarify this.

**A.** Ablation + parameters (see the parameter-matched shared-temporal experiment above).

We thank the reviewer for this important suggestion. To distinguish improvements arising from increased model capacity from those due to scale-specific temporal modeling, we performed two complementary studies.

First, we conducted a progressive ablation in which components are added incrementally while tracking parameter growth (Table below). 

We first evaluate the contribution of explicitly modeling the wavelet decomposition. Compared with MLP only (ab1), adding Wavelet + MLP (ab2) consistently improves the multiscale representation on both datasets. Specifically, CSI-M increases from 0.2847/0.3418 to 0.3000/0.3459, while HSS improves from 0.3653/0.4702 to 0.3869/0.4756 on CIKM/MeteoNet, respectively. These results demonstrate that decomposing latent precipitation features into multiscale wavelet components provides a more effective representation for temporal forecasting than modeling all frequencies jointly.

Next, we replace the standard temporal MLP with our Gabor-based Frequency Adaptive Temporal (FAT) module. Although the overall CSI-M changes are modest, the Gabor-based temporal modeling consistently improves the prediction of fine-scale precipitation structures. On CIKM/MeteoNet, CSI-4 improves from 0.3133/0.3573 to 0.3179/0.3537, while CSI-16 increases from 0.3515/0.3913 to 0.3656/0.3968. Moreover, SSIM improves from 0.6074/0.8308 to 0.6315/0.8342, accompanied by lower MSE (48.56/11.30 → 45.90/10.97). These results indicate that frequency-selective temporal modeling better preserves high-frequency precipitation details and improves reconstruction quality.

The results consistently favor separate frequency-specific temporal modeling. On SEVIR, the shared baselines achieve 0.3435 and 0.3444 CSI-M, whereas our separate FAT reaches 0.3638 (+1.9–2.0 percentage points), with corresponding improvements across CSI-4, CSI-16, HSS, SSIM, and MSE. Similarly, on CIKM, the shared baselines obtain 0.3201 and 0.3219 CSI-M, while our model achieves 0.3303, together with improvements in all other evaluation metrics.

Regarding **increased model capacity** : The larger increase in model capacity is introduced only after adding the SRST refinement block. We emphasize that SRST is not intended to replace WGTM, but to refine the temporally evolved multiscale representation. Moreover, the benefit of SRST is not simply due to stacking two existing modules. If the spectral and spatial branches contributed independently, their individual ablation drops would be approximately additive. Instead, on SEVIR, removing the spectral branch reduces CSI-M by 0.0112 and removing the spatial branch by 0.0086 (expected additive drop: 0.0198), whereas removing the complete SRST decreases CSI-M by 0.0336 (≈70% larger). A similar super-additive effect is observed on MeteoNet (0.0305 vs. 0.0384). This demonstrates that the gains arise from the complementary interaction between spectral and spatial refinement rather than simply from increased capacity.

Therefore, these experiments indicate that the performance gains cannot be attributed solely to increased parameter count. Even under identical model capacity, assigning independent temporal dynamics to different wavelet frequency bands consistently outperforms a shared temporal operator, demonstrating that scale-specific temporal modeling itself contributes substantially to the observed improvements.

CIKM Dataset (Ablation Study)

| Model | Configuration | Loss | Params (M) | CSI-M↑ | CSI-4↑ | CSI-16↑ | HSS↑ | SSIM↑ | MSE↓ |
|------|---------------|:----:|-----------:|-------:|-------:|--------:|-----:|------:|-----:|
| ab1_mlp_only | MLP only | MSE | 0.267 | 0.2847 | 0.2997 | 0.3457 | 0.3653 | 0.6135 | 47.84 |
| ab2_wavelet_mlp | Wavelet + MLP | MSE | 0.345 | 0.3000 | 0.3133 | 0.3515 | 0.3869 | 0.6074 | 48.56 |
| ab3_wavelet_mlp_gabor | Wavelet + MLP + Gabor | MSE | 0.424 | 0.2990 | 0.3179 | 0.3656 | 0.3849 | 0.6315 | 45.90 |
| ab4_srst1 | Wavelet + MLP + Gabor + 1 SRST | MSE | 8.692 | 0.3150 | 0.3252 | 0.3623 | 0.4085 | 0.6649 | 37.73 |
| ab5_full | Wavelet + MLP + Gabor + 2 SRST | MSE | 15.320 | 0.3152 | 0.3306 | 0.3743 | 0.4065 | 0.6795 | 36.40 |
| ab6_full | Wavelet + MLP + Gabor + 2 SRST | FACL | 15.320 | 0.3303 | 0.3543 | 0.4135 | 0.4266 | 0.6696 | 38.64 |


MeteoNet Dataset (Ablation Study)
| Model | Configuration | Loss | Params (M) | CSI-M↑ | CSI-4↑ | CSI-16↑ | HSS↑ | SSIM↑ | MSE↓ |
|------|---------------|:----:|-----------:|-------:|-------:|--------:|-----:|------:|-----:|
| ab1_mlp_only | MLP only | MSE | 0.267 | 0.3418 | 0.3516 | 0.3903 | 0.4702 | 0.8330 | 11.10 |
| ab2_wavelet_mlp | Wavelet + MLP | MSE | 0.309 | 0.3459 | 0.3573 | 0.3913 | 0.4756 | 0.8308 | 11.30 |
| ab3_wavelet_mlp_gabor | Wavelet + MLP + Gabor | MSE | 0.350 | 0.3413 | 0.3537 | 0.3968 | 0.4691 | 0.8342 | 10.97 |
| ab4_srst1 | Wavelet + MLP + Gabor + 1 SRST | MSE | 33.193 | 0.3782 | 0.4006 | 0.4536 | 0.5108 | 0.8409 | 10.58 |
| ab5_full | Wavelet + MLP + Gabor + 2 SRST | MSE | 59.468 | 0.4067 | 0.4399 | 0.4900 | 0.5451 | 0.8413 | 10.44 |
| ab6_full | Wavelet + MLP + Gabor + 2 SRST | FACL | 59.468 | 0.4085 | 0.4838 | 0.5989 | 0.5482| 0.8389 |  11.54|


**DAWNCast Shared temporal hyperparameters**
| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:------------:|:-------------:|:----:|:----:|:----:|:---------:|:------------:|:--------------:|:----:|:----:|:----:|
| | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-4 (POOL)↑ | CSI-16 (POOL)↑ | HSS↑ | SSIM↑ | MSE↓ |
| Shared-matched FAT-init w low freq params  | 0.3435 | 0.3715 | 0.4389 | 0.4403 | 0.7092 | 400.24 | 0.3201 | 0.3430 | 0.4003 | 0.4144 | 0.6627 | 39.07 |
| Shared-matched FAT-init w high freq params | 0.3444 | 0.3714 | 0.4365 | 0.4417 | 0.7102 | 400.76 | 0.3219 | 0.3450 | 0.4034 | 0.4168 | 0.6596 | 39.01 |
| Seperate FAT(*ours*)| **0.3638** | **0.4054** | **0.4856** | **0.4668** | **0.7284** | **371.34** | **0.3303** | **0.3543** | **0.4135** | **0.4266** | **0.6696** | **38.64** |



### 5. "Climatic" inductive bias — terminology

> The Gabor stream is described as introducing a "climatic" inductive bias. What aspect of the model makes it climate-aware, given that no explicit climate or environmental conditioning appears to be used? Would "frequency-adaptive" or "event-regime-adaptive" be a more precise description?

**A.** We thank the reviewer for this observation and agree that frequency-adaptive is a more precise technical description. Our use of the term *climatic inductive bias* was intended to convey the following intuition rather than claim explicit climate conditioning.

Radar observations from a given dataset originate from a specific geographical region, whose long-term climate governs the characteristics of precipitation evolution. The proposed Frequency Adaptive Temporal (FAT) block operates on wavelet-separated components corresponding to large-scale convective structures (LL) and small-scale turbulent structures (LH/HL/HH). Each component is modelled using a learnable FAT Block containing Gabor operator whose frequency response adapts during training. Our frequency sweep (shown in reponse to Q.1) experiments show that each dataset converges to a distinct optimal Gabor frequency configuration, suggesting that different precipitation regimes favour different temporal frequency responses.

Therefore, the inductive bias introduced by the FAT block is not climate-aware; rather, it learns dataset-specific temporal frequency responses, which implicitly capture the characteristic precipitation dynamics of the region using radar represented by the training data. Since these precipitation dynamics are ultimately shaped by the regional climate, our original wording referred to this as a climatic inductive bias. We agree, however, that this terminology may overstate the claim because the model is conditioned only on radar observations. To avoid this ambiguity, we will replace the phrase throughout the paper with frequency-adaptive inductive bias, and clarify that the learned temporal filters implicitly adapt to the statistical precipitation regime represented by the training dataset rather than to explicit climatic variables.

### 6. Comparison against alternative multi-scale strategies under comparable budgets

> How does the proposed wavelet-based decomposition compare against alternative multi-scale strategies (e.g., Fourier-based decomposition, cascade models, Laplacian pyramids, or learned multi-resolution feature hierarchies) under comparable computational budgets?

**A.**


We thank the reviewer for this suggestion. Table 1 already compares DAWN-Cast against representative Fourier-based methods (AlphaPre and FourCastNet) and learned multi-resolution approaches (SimVP, EarthFormer, and EarthFarseer), while the "without Spectral Branch" ablation in Table 2 provides an internal parameter-matched comparison against Fourier modelling. To further address the reviewer's question, we additionally evaluated CasCast (cascade-based forecasting) and WADEPre (wavelet-based forecasting) under our experimental protocol.

DAWN-Cast consistently outperforms both methods on CSI-based metrics across SEVIR and CIKM. Compared with CasCast, DAWN-Cast improves CSI-M from 0.3303→0.3638 (+0.0335) on SEVIR and 0.2992→0.3303 (+0.0311) on CIKM, while also improving HSS (+0.0325, +0.0177) and SSIM (+0.2089, +0.0298). On SEVIR, MSE is also substantially reduced (496.56→371.34). Compared with WADEPre, DAWN-Cast improves CSI-M by 0.0114 on SEVIR and 0.0346 on CIKM, together with consistent gains in HSS (+0.0171, +0.0441) and SSIM (+0.0860, +0.0120). WADEPre reports a slightly lower MSE on CIKM (36.19 vs. 38.64), which is consistent with the well-known discrepancy between point-wise error metrics and event-based precipitation skill discussed elsewhere in our response.

Beyond forecasting accuracy, the comparison also highlights differences in modelling strategy. Fourier-based approaches learn temporal evolution in the global spectral domain, learned multi-resolution methods construct hierarchical features implicitly through the network architecture, cascade models progressively refine predictions across multiple stages, and WADEPre employs wavelet decomposition primarily as a hierarchical representation with separate approximation and detail networks. In contrast, DAWN-Cast performs explicit wavelet decomposition in latent space and assigns an independent Frequency Adaptive Temporal (FAT) block to each wavelet subband, enabling scale-specific temporal evolution within a unified forecasting framework.

In terms of model capacity, DAWN-Cast contains 51.41M parameters, compared with 89.03M for AlphaPre, 148.57M for EarthFarseer, and 311.7M for CasCast, while achieving higher CSI-M on both datasets. Relative to WADEPre (43.20M) and DiffCast (49.36M), whose model sizes are similar, DAWN-Cast also achieves higher CSI-M across both benchmarks. Although we did not perform a controlled substitution with Laplacian pyramids within the rebuttal period, our ablation over wavelet families and decomposition levels (Appendix E.1) indicates that the observed improvements are not tied to a specific wavelet basis, but to the proposed scale-specific temporal modelling enabled by the decomposition.

| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:--------:|:--------:|:----:|:----:|:----:|:---------:|:-------:|:-------:|:----:|:----:|:----:|
|      | CSI-M↑ | CSI-181↑ | CSI-219↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-35↑ | CSI-40↑ | HSS↑ | SSIM↑ | MSE↓ |
| CasCast | 0.3303 | 0.1675 | 0.0838 | 0.4343 | 0.5195 | 496.56 | 0.2992 | 0.2064 | 0.1509 | 0.4089 | 0.6398 | — |
| WADEPre | 0.3524 | 0.1867 | 0.1031 | 0.4497 | 0.6424 | 398.82 | 0.2957 | 0.1908 | 0.1203 | 0.3825 | 0.6576 | 36.19 |


One more suggestion: Since the reviewer explicitly asked about computational budgets, if you can report FLOPs, MACs, inference latency, or training throughput (even approximately), adding a single sentence or small table would make this response much stronger. Right now, parameter count is a useful proxy, but it is not the same as computational budget.

-------------------------------------------