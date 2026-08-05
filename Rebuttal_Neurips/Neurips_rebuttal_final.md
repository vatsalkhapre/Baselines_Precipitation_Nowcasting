

# Reviewer 1

### 1. Wavelet transforms are common; add discussion and comparisons

> Utilizing wavelet transforms for multi-scale separation is relatively common in the fields of meteorological forecasting and image processing (e.g., Reference [1]). The authors should supplement the paper with relevant discussions and comparative experiments regarding these prior works.

**A.**

We appreciate the reviewer's observation that wavelet transforms are a well-established for multi-scale signal decomposition in meteorological forecasting and image processing. However, **we emphasize that the novelty of DAWN-Cast does not lie in the use of the wavelet transform, but in how the resulting decomposition is exploited for temporal modelling**. Our analysis of precipitation dynamics from a multi-scale perspective demonstrates that the 2D discrete wavelet decomposition separates the latent precipitation field into components with markedly different characteristics: the **low-frequency (LL)** component captures the slowly evolving **large-scale precipitation structure**, whereas the **high-frequency (LH/HL/HH)** components represent localized **fine-scale variability**, as shown in (Fig 1 & 5). This empirical observation motivates the proposed architecture, in which each wavelet subband is assigned a dedicated Frequency Adaptive Temporal (FAT) block, enabling the model to learn scale-specific temporal dynamics rather than relying on a shared temporal operator across all frequency bands.

**Comparision with WADEPre (ArXiv Feb 2026)**

The Wavelet based method "WADEPre (ArXiv Feb 2026)" recommended by the reviewer is a very recent relevant work, which addresses the dual challenges of blurry extremes and spatial localization errors through wavelet-based disentanglement and a stable coarse-to-fine curriculum learning framework. We have compared against this Model under our evaluation protocol (Table 1 below): 

Table 1: 
| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:------:|:------:|:----:|:----:|:----:|:-----------:|:------:|:------:|:----:|:----:|:----:|
|      | CSI-M↑ | CSI-181↑ | CSI-219↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-35↑ | CSI-40↑ | HSS↑ | SSIM↑ | MSE↓ |
| WADEPre | 0.3524 | 0.1867 | 0.1031 | 0.4497 | 0.6424 | 398.82 | 0.2957 | 0.1908 | 0.1203 | 0.3825 | 0.6576 | **36.19** |
| DAWN-Cast (*ours*)| **0.3638** | **0.1950** | **0.1077** | **0.4668** | **0.7284** | **371.34** | **0.3303** | **0.2349** | **0.1591** | **0.4266** | **0.6696** | 38.64 |

As shown above, DAWN-Cast consistently outperforms WADEPre across the majority of forecasting metrics.

WADEPre's major limitation is that, it is naturally designed for equal input/output horizons (6→6 in their experiments), whereas our architecture directly supports arbitrary forecasting horizons without modifying the temporal modelling mechanism. 

Although both DAWN-Cast and WADEPre employ a Discrete Wavelet Transform, the role of the wavelet decomposition is fundamentally different. In WADEPre, the decomposition primarily serves as a representation mechanism: the approximation coefficients are processed by an Approximation Network while the detail coefficients are processed by a Detail Network before being fused through a refinement stage.In contrast, our motivation is not simply to process different wavelet bands separately, but to model their temporal evolution differently. We hypothesize that the LL component and the LH/HL/HH components exhibit different characteristics, as supported by our wavelet statistics (Fig. 5 and Appendix C). Consequently, each subband is assigned an independent Frequency Adaptive Temporal (FAT) block, allowing the model to learn distinct temporal dynamics for each frequency regime rather than relying on a shared temporal operator.

Although this does not affect the proposed methodology, we will include a brief discussion of wavelet based methods in the revised manuscript for completeness.


### 2. SimVP already captures multi-scale structure implicitly

> The claim (Line 54) that existing latent-space methods lack physical scale structures is inaccurate. In reality, methods such as SimVP implicitly capture and utilize multi-scale structural information through their downsampling and upsampling network architectures. The authors need to clarify the fundamental differences between their approach and these existing methods.

> **Note:** the claim at Line 54 relates to claim 3; see how it can be linked.

**A.** 

We thank the reviewer for raising this point. We believe there may be a misunderstanding regarding the referenced statement. The cited Line 54 ("existing latent-space methods compress radar...") does not discuss multi-scale representations. Rather, our statement regarding multi-scale structure appears earlier (Line 46): "most approaches do not explicitly exploit the multiscale structure of precipitation." 

Methods such as SimVP indeed learn hierarchical multi-resolution features through encoder-decoder downsampling and upsampling. Our distinction is that these hierarchies are implicitly learned feature representations, whereas DAWN-Cast employs an explicit, interpretable wavelet decomposition whose subbands are subsequently assigned dedicated temporal models.

Specifically, three properties distinguish our approach:

(i) **Invertibility.** The Discrete Wavelet Transform (DWT) is a perfect-reconstruction transform, allowing individual subbands to be processed independently and reconstructed exactly. In contrast, encoder-decoder downsampling via strided convolutions is inherently lossy and does not provide an explicit decomposition into identifiable scales.

(ii) **Addressability.** The wavelet subbands have explicit semantic meaning: the LL component corresponds to large scale precipitation structure, while the LH/HL/HH components capture fine-scale spatial variability. Our wavelet analysis (Fig. 5 and Appendix C) further suppports that the LL component exhibits substantially higher temporal persistence than the high-frequency subbands, motivating independent temporal modelling. In contrast, the feature channels learned by SimVP do not possess an explicit scale interpretation.

(iii) **Scale-specific temporal modelling.** Existing implicit multiscale decomposition methods apply a shared temporal operator across the learned feature hierarchy. In DAWN-Cast, each wavelet subband is assigned an independent Frequency Adaptive Temporal (FAT) block, enabling the model to learn distinct temporal dynamics for different spatial-frequency regimes. Thus, the wavelet decomposition serves not merely as a representation, but as the basis for scale-specific temporal modelling.

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

------


We would like to clarify that the proposed WGTM is not intended to replace the refinement backbone, but to provide an explicit multiscale temporal inductive bias before refinement. By decomposing the latent representation into wavelet subbands and assigning each subband an independent Frequency Adaptive Temporal (FAT) block, the model learns scale-specific temporal evolution before the SRST reconstructs globally consistent latent features.

This behaviour is consistently supported by the ablation studies (Table 2). Removing the wavelet decomposition, removing the Gabor/FAT module, or replacing the adaptive with non adaptive gabors (Table 3) or using FAT blocks with identical parameter initializations for all scales(Table above in weakness 2) all reduce performance across datasets. These results indicate that explicitly modelling the temporal dynamics of different wavelet subbands using specialized temporal operator (FAT block), supporting the contribution of the proposed WGTM. The Gabor stream is not additive — the WGTM depends on it. Table 2 shows that removing the Gabor stream while retaining the WGTM (SEVIR 0.3541) is worse than removing the entire WGTM block (0.3566). The ordering replicates on MeteoNet (0.4002 vs 0.4054).

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


**We thank the reviewer for this constructive wonderful review on this paper.**
======================================================================================================================================================================================================================


# Reviewer 2

> "For example, if a physical inconsistency would be visible from the SOTA but instead it only talks about technical things." … "…but this sounds like the MSE should be improved which then does not happen."

**A.** 

We totally understand your point and wish to refine and increase the current research gap section in revised version by including the visible inconsistency problems as well like blurring, position inconsistency, inconsistency over lead times, retaination of high intensity over larger lead times and other similar visible inconsistent factors across various models, and also how our model is trying to improve on any of these inconsistencies. 

To further clarify our model's motivation, the objective of explicitly exploiting the multi-scale structure is to better model the evolution of precipitation systems. By decomposing radar observations into multiple spatial scales, the model can more effectively capture the dynamics of both large-scale precipitation bands and finer-scale convective structures, leading to more accurate precipitation nowcasting, <as highlighted by previous papers previously>. Hence we do explicit multiscale decomposition of the radar observations and notice that they correspond to characteristics which can be correlated with actual precipitation structure characteristics as stated in the contribution 1 statement "low-frequency values correspond to the large-scale structure and high-frequency values correspond to the fine-scale variability of the precipitation field.".  We observe different precipitation field characteristics pose different motion trends which we use as motivation for building different specialized gabor based FAT blocks, and we notice that sinosoids in gabor can help us do that. Sinosoids can vary from near linear(small z in sin(z)) to oscillatory trend (large z). We also show in Table 4. how different gabor configurations help to model different situations effectively, technically giving us manual ability to have a best model for different atmospheric dynamics (Ref Table 4). 

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

This is a minor edit, we will insert the recommended citations (Gabor, CSI, HSS, autoformers) in the revised version paper. 

[1] Roberts, Nigel M., and Humphrey W. Lean. "Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events." Monthly Weather Review 136.1 (2008): 78-97.


# Questions 

> Q. What is the motivation for developing an architecture like this other than the idea of it being closer to physics should make it unspecifically better?

**A** 

Our motivation is not that a more physics-inspired architecture is inherently better. Rather, we hypothesize that explicitly separating precipitation into components with different characteristics(large salce convective bulk and fine scale variability) simplifies the learning problem in approximation of precipitation field movement through radar observations. 

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


> **Limitations**: The authors claim that this could be used for other problems of Computervision (line 321) but i cannot see how that is true.

**A** Although our original statement was broad, our intention was not to claim applicability to all computer vision problems, but rather to certain spatiotemporal prediction tasks that share similar characteristics with precipitation nowcasting, namely the forecasting of structured spatial fields evolving over time. Examples include satellite imagery forecasting, remote sensing, dynamic medical imaging (e.g., cardiac MRI or ultrasound sequence prediction), and related scientific imaging applications. This is a minor edit, and we will make this change in the revised manuscript by explicitly stating these application domains and avoiding the broader term *computer vision*.


======================================================================================================================================================================================================================


# Reviewer 3

### 1. Main conceptual novelty vs cascade/spectral/multi-resolution priors

> Multi-scale and cascade-based methods are widely studied in precipitation nowcasting. What is the main conceptual novelty of DAWN-Cast compared to prior cascade, spectral, or multi-resolution approaches? Is the key contribution the wavelet decomposition, the latent-space formulation, the FAT block, or their integration?

**A.** 

The conceptual novelty of DAWN-Cast is the Wavelet Guided Temporal Modelling (WGTM) block, which factorizes the latent sequence into low- and high-frequency wavelet subbands, each assigned a dedicated Frequency Adaptive Temporal (FAT) block to learn scale-specific temporal dynamics. 

Our modelling objective also differs from prior families of approaches as indicated below:

1. Fourier-based (AlphaPre, Fourcastnet): AlphaPre decomposes by physical quantity (intensity and motion); FourCastNet doesn't decompose at all, instead it globally mixes the spectrum. Neither separates by scale, which is DAWN-Cast's axis.
2. Multi-resolution (SimVP and Earthformer):  Scales exist as features but are never separated into independently modelled channels, and the pyramid isn't invertible.
3. Cascade (Cascast and Diffcast): Sequential conditioning on a prior stage's residual vs. parallel branches
4. Wavelet (WADEPre): DWT used spatially for reconstruction quality; temporal treatment identical across subbands

To support our claimed novelty, refer to additional analyses: Table 3 and Table 4.

In order to futher support the evidence we perfomed additional experiments shown below:
1. Introducing same hyperparameter initialized FAT blocks for all the scales also degrades the performance. Shown in the Table 1 of **Question response of Reviewer ULBe**, named *DAWNCast shared temporal hyperparameters*.
2. We swept the Gabor frequency multiplier λ jointly across the LL and HF FAT blocks, shown in Table 1 below. Note that λ is not the sine's argument: the true argument is θ = λ·f·(W·x+b), θ behavior can be referred from Sec. 2.3. Five λ corresponds to five θ that spans from near-linear to a full half-cycle (θ≈π) in a regular interval.

Table 1:
CIKM frequency sweep CSI-M as a function of (λ_LL, λ_HF)
| HF \ LL | 22.74 | 68.23 | 181.94 | 363.89 | 714.49 |
|:-------:|:-----:|:-----:|:------:|:------:|:------:|
| **95.56** | 0.3269 | 0.3292 | 0.3201 | 0.3289 | 0.3264 |
| **48.67** | **0.3362 🥇** | 0.3235 | 0.3224 | 0.3217 | 0.3252 |
| **24.34** | 0.3276 | 0.3243 | 0.3292 | 0.3246 | 0.3248 |
| **9.13**  | 0.3275 | 0.3296 | 0.3256 | 0.3237 | 0.3273 |
| **3.04**  | 0.3266 | *0.3311 🥈* | 0.3253 | 0.3270 | 0.3234 |

Meteonet frequency sweep CSI HEATMAP: : **Kindly Refer to the Weakness response of Reviewer ULBe**

### 2 & 3. Stronger evidence that latent subbands correspond to physical radar structures

> Since wavelet decomposition is applied in latent space, can the authors provide stronger evidence that low-/high-frequency subbands correspond to physically meaningful precipitation structures in the original radar field?
> Are the autocorrelation and spectral analyses performed on the original radar fields, latent representations, or both? If primarily on original fields, how is the physical interpretation transferred to the latent-space decomposition?

**A**

Our model operates in the latent space of the per-dataset autoencoder introduced by Rombach et al. [1]. Owing to its convolutional, strided encoder, the latent representation preserves spatial correspondence with the pixel representation, a property further analyzed by Bradbury and Zhong [2], who show that latent positions remain spatially aligned with their corresponding image regions. Consequently, the latent features preserve the spatial precipitation structures of the original radar observations.

Our original analyses in Figures 1 and 5 were intentionally performed in pixel space to visually demonstrate that the wavelet low- and high-frequency subbands correspond to large-scale structures and fine-scale variability of precipitation. To address the reviewer's concern, we analyzed the latent representations and compared them with the original radar fields using the required metrics (Table below).

Table 1:
| Dataset (n) | Field | Pixel (ρ) | Latent (ρ) | LFE (pix/lat) | PSD-shape r |
|:-----------|:------|----------:|-----------:|:-------------:|------------:|
| **CIKM** | GT – raw radar field | 0.959 | 0.562 | 0.999 / 0.986 | 0.983 |
|  | LL | 0.939 | 0.573 | 0.998 / 0.971 | 0.990 |
|  | HF | 0.130 | 0.067 | 0.494 / 0.319 | 0.564 |
| **MeteoNet** | GT – raw radar field | 0.895 | 0.562 | 0.995 / 0.972 | 0.943 |
|  | LL| 0.791 | 0.591 | 0.984 / 0.965 | 0.992 |
|  | HF| -0.235 | -0.047 | 0.083 / 0.198 | 0.799 |
| **Shanghai** | GT – raw radar field | 0.947 | 0.652 | 0.998 / 0.975 | 0.945 |
|  | LL | 0.880 | 0.674 | 0.993 / 0.955 | 0.987 |
|  | HF | -0.248 | -0.049 | 0.074 / 0.209 | 0.893 |
| **SEVIR** | GT – raw radar field | 0.937 | 0.566 | 0.999 / 0.968 | 0.926 |
|  | LL | 0.859 | 0.611 | 0.994 / 0.958 | 0.992 |
|  | HF | -0.242 | -0.042 | 0.084 / 0.206 | 0.542 |

ρ : Lag-1 autocorrelation.
LFE: Fraction of spectral power contained in lowest-quarter wavenumbers.
PSD-shape r: Pearson correlation (log-space) between pixel and latent PSD curves.

The table reports the same statistics computed directly on the latent tensor Z, showing that the latent subbands preserve the large-scale/fine-scale distinction observed in pixel space.

Reference: 
1. Rombach, R.,et. al, 2022. High-resolution image synthesis with latent diffusion models. CVPR.
2. Bradbury, R.et. al, 2025. Your Latent Mask is Wrong: Pixel-Equivalent Latent Compositing for Diffusion Models. arXiv preprint.


### 4. Scale-specific temporal modeling vs increased capacity

> To what extent do improvements come from true scale-specific temporal modeling versus increased model capacity? A parameter-matched shared-temporal baseline would help clarify this.

**A.** 

To distinguish improvements arising from increased model capacity from those due to scale-specific temporal modeling, we performed two complementary studies.

First, we conducted a progressive ablation in which components are added incrementally while tracking parameter growth for 2 datasets. 

Ablation Study: 

CIKM Dataset 
| Added Component | Loss | Params (M) | CSI-M↑ | CSI-4↑ | CSI-16↑ | HSS↑ | SSIM↑ | MSE↓ |
|:---------------|:----:|-----------:|-------:|-------:|--------:|-----:|------:|-----:|
| MLP | MSE | 0.267 | 0.2847 | 0.2997 | 0.3457 | 0.3653 | 0.6135 | 47.84 |
| + Wavelet | MSE | 0.345 | 0.3000 | 0.3133 | 0.3515 | 0.3869 | 0.6074 | 48.56 |
| + Gabor FAT | MSE | 0.424 | 0.2990 | 0.3179 | 0.3656 | 0.3849 | 0.6315 | 45.90 |
| + 1 SRST | MSE | 8.692 | 0.3150 | 0.3252 | 0.3623 | 0.4085 | 0.6649 | 37.73 |

MeteoNet Dataset 
| Added Component | Loss | Params (M) | CSI-M↑ | CSI-4↑ | CSI-16↑ | HSS↑ | SSIM↑ | MSE↓ |
|:---------------|:----:|-----------:|-------:|-------:|--------:|-----:|------:|-----:|
| MLP | MSE | 0.267 | 0.3418 | 0.3516 | 0.3903 | 0.4702 | 0.8330 | 11.10 |
| + Wavelet | MSE | 0.309 | 0.3459 | 0.3573 | 0.3913 | 0.4756 | 0.8308 | 11.30 |
| + Gabor FAT | MSE | 0.350 | 0.3413 | 0.3537 | 0.3968 | 0.4691 | 0.8342 | 10.97 |
| + 1 SRST | MSE | 33.193 | 0.3782 | 0.4006 | 0.4536 | 0.5108 | 0.8409 | 10.58 |

Table 1 of paper shows the improvement of FACL over MSE. 

Second, to isolate the effect of scale-specific temporal modelling from model capacity, we constructed a parameter-matched baseline by sharing a same FAT block across all wavelet subbands, keeping the overall parameters matched, scores reported in **Table 1 of Question response of Reviewer ULBe, named DAWNCast shared temporal hyperparameters**, showing that parameter-matched shared-FAT baseline consistently underperforms. In addition, the parameter-matched MLP scores are reported in Table 2 of the paper.

Regarding increased model capacity, the substantial increase in parameters occurs only after the SRST module, whose contribution arises from complementary spectral-spatial interactions, (**for more clarification refer to Weakness 3 of reviewer bUgb**). Parameter comparison with other baselines in Q.6(Table 1).


### 5. "Climatic" inductive bias — terminology

> The Gabor stream is described as introducing a "climatic" inductive bias. What aspect of the model makes it climate-aware, given that no explicit climate or environmental conditioning appears to be used? Would "frequency-adaptive" or "event-regime-adaptive" be a more precise description?

**A.** Our use of the term *climatic inductive bias* was intended to convey the following intuition:

Radar observations from a given dataset originate from a specific geographical region, whose long-term climate governs the characteristics of precipitation evolution. The proposed Frequency Adaptive Temporal (FAT) block operates on wavelet-separated components corresponding to large-scale convective structures (LL) and small-scale turbulent structures (LH/HL/HH). Each component is modelled using a learnable FAT Block containing Gabor operator whose frequency response learns during training. Our frequency sweep (shown in reponse to Q.1) experiments show that each dataset converges to a distinct optimal Gabor frequency configuration, suggesting that different precipitation regimes favour different temporal frequency responses.

Therefore, the inductive bias introduced by the FAT block is not explicity climate-aware; rather, it learns dataset-specific temporal frequency responses, which implicitly capture the characteristic precipitation dynamics of the region using radar represented by the training data. Since these precipitation dynamics are ultimately shaped by the regional climate, our original wording referred to this as a climatic inductive bias. We agree, however, that this terminology may overstate the claim because the model is conditioned only on radar observations. To avoid this ambiguity, we will replace the phrase throughout the paper with frequency-adaptive inductive bias, and clarify that the learned temporal filters implicitly adapt to the statistical precipitation regime represented by the training dataset rather than to explicit climatic variables.

### 6. Comparison against alternative multi-scale strategies under comparable budgets

> How does the proposed wavelet-based decomposition compare against alternative multi-scale strategies (e.g., Fourier-based decomposition, cascade models, Laplacian pyramids, or learned multi-resolution feature hierarchies) under comparable computational budgets?

**A.**

Table 1 of paper already compares DAWN-Cast with representative Fourier-based, cascade and learned multi-resolution methods. To further address the reviewer's question, we additionally evaluated CasCast (cascade-based) and WADEPre (wavelet-based) under the same experimental protocol.

The differences between existing fourier-based, cascade, learned multi-resolutionand wavelet based approaches are *discussed in Q.1 above*. 
Table 1:
| Model | **SEVIR** |  |  |  |  |  | **CIKM** |  |  |  |  |  |
|------|:---------:|:--------:|:--------:|:----:|:----:|:----:|:---------:|:-------:|:-------:|:----:|:----:|:----:|
|      | CSI-M↑ | CSI-181↑ | CSI-219↑ | HSS↑ | SSIM↑ | MSE↓ | CSI-M↑ | CSI-35↑ | CSI-40↑ | HSS↑ | SSIM↑ | MSE↓ |
| CasCast | 0.3303 | 0.1675 | 0.0838 | 0.4343 | 0.5195 | 496.56 | 0.2992 | 0.2064 | 0.1509 | 0.4089 | 0.6398 | 41.81 |
| WADEPre | 0.3524 | 0.1867 | 0.1031 | 0.4497 | 0.6424 | 398.82 | 0.2957 | 0.1908 | 0.1203 | 0.3825 | 0.6576 | 36.19 |

For completeness, we also provide a comparison of model complexity (parameters, FLOPs, and inference throughput) for recent representative methods. 
Table 2:
| Model | Parameters (M) | FLOPs (G) | Throughput (samples/s) |
|:------|---------------:|----------:|-----------------------:|
| AlphaPre | 89.03 | 1556.67 | 6.6 |
| EarthFarseer | 148.57 | 753.32 | 8.7 |
| EarthFormer | 8.65 | 35.24 | 82.9 |
| DiffCast | 49.36 | 30506.49 | 0.1 |
| CasCast | 311.77 | 51.07 | 1.72 |
| WADEPre | 24.99 | 1274.92 | 6.6 |
| DAWNCast (latent) | 51.41 | 58.74 | 50.9 |

Although a controlled comparison with Laplacian pyramids was beyond the rebuttal period, they share similar properties with wavelet but are typically overcomplete (≈4/3× coefficient redundancy) rather than critically sampled, making them an interesting direction for future comparison.


**We thank the reviewer for this constructive wonderful review on this paper.** 
======================================================================================================================================================================================================================