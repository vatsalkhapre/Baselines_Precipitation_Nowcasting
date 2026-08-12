> The main concern is limited conceptual novelty. Wavelet decomposition, spectral modeling, multi-scale processing, Gabor activations, and global-local refinement are all established ideas. The paper does not clearly identify which component constitutes the main methodological advance beyond their integration.

**A.** 

The conceptual novelty of DAWN-Cast is not the individual use of wavelets, Gabor operators, or multi-scale processing, all of which are established. Rather, the methodological contribution is the proposed Wavelet Guided Temporal Modelling (WGTM) block, which factorizes the latent representation into low- and high-frequency wavelet subbands and assigns each an independent Frequency Adaptive Temporal (FAT) block to learn scale-specific temporal dynamics. Unlike prior Fourier-, cascade-, multi-resolution-, or wavelet-based approaches, the wavelet decomposition is used to determine how temporal evolution is modelled. To support this distinction, we provided additional analyses, including comparisons with recent wavelet-based methods, shared-versus-independent FAT modelling, and adaptive Gabor frequency sweeps demonstrating that different subbands and datasets converge to different temporal frequency responses.

**Please see our responses to Reviewer NKQs (Q1), Reviewer bUgb (W1–W2) for the detailed discussion and supporting experiments.**

>The ablation results also weaken the central claim. Removing the SRST block causes the largest performance drop, although this block mainly combines existing AFNO-style processing with depthwise convolution. In contrast, removing the Gabor stream produces only modest degradation. This suggests that the main gains may not come from the proposed frequency-adaptive temporal mechanism.

**A.** 
The proposed WGTM and SRST blocks serve complementary roles rather than competing ones. WGTM introduces the paper's main methodological contribution by explicitly decomposing the latent representation into wavelet subbands and assigning each an independent Frequency Adaptive Temporal (FAT) block for scale-specific temporal modelling, while SRST reconstructs globally consistent latent features after temporal evolution. Our additional analyses show that removing the wavelet decomposition, the adaptive FAT/Gabor component, or replacing independent FAT blocks with shared temporal modelling consistently degrades performance, supporting the contribution of the proposed temporal mechanism. We further demonstrate that the effectiveness of SRST arises from the complementary interaction between its spectral and spatial refinement streams rather than simply combining two existing modules. Finally, the parameter-matched shared-temporal baseline confirms that the observed improvements arise from explicit scale-specific temporal modelling rather than increased model capacity.

**Please see our responses to Reviewer bUgb (W3) and Reviewer NKQs (Q4) for the detailed ablations, parameter-matched comparisons, and additional analyses.**

>The physical interpretation is not sufficiently supported because wavelet decomposition is applied to learned latent features rather than directly to radar fields. It is therefore unclear whether the resulting subbands correspond to physically meaningful precipitation scales. The “climatic inductive bias” claim is also overstated, since the model does not use climate labels or environmental variables.

**A.**
In order to support the physical interpretation of the latent-space wavelet decomposition, we provided new analyses in response to Reviewer NKQs (Q2 and Q3) comparing the latent representations with the original radar fields using spatial autocorrelation, low-frequency spectral energy, and power spectral density statistics. These results show that the latent subbands preserve the large-scale/fine-scale distinction observed in the original radar fields, supporting our interpretation of the latent-space wavelet decomposition. In addition, to the analyses paragraph 1  clarifies the spatial correspondence between the latent and pixel representations, providing the basis for relating the latent-space analysis to the original radar fields.

**Please see our response to Reviewer NKQs (Q2 and Q3) for the detailed analyses.**

Regarding the terminology, the concern regarding "climatic inductive bias" has been addressed in our response to Reviewer NKQs (Q5), where we clarify the intended meaning.


> The evaluation does not fully establish the proposed mechanism. Parameter-matched baselines, alternative decompositions, and comparisons between shared and subband-specific temporal modeling are missing. Part of the improvement may also come from the FACL loss rather than the architecture. Moreover, the model does not improve MSE consistently, despite the motivation involving intensity errors at longer lead times.

**A.** We believe the additional analyses directly address these concerns. To distinguish improvements arising from scale-specific temporal modelling rather than model capacity, we introduced parameter-matched baselines, including a shared-FAT baseline(in the response to reviewer) and a parameter-matched MLP baseline(Table 2 of paper, w/o Gabor), showing that shared temporal modelling consistently underperforms independent subband-specific temporal modelling. Contribution coming from FACL loss alone can be seen from Table 1 of the paper. 

We give a clarification regarding MSE by discussing the limitations of point-wise error for precipitation nowcasting and providing additional analyses showing that lower MSE does not necessarily correspond to higher CSI under different temporal frequency configurations.

**Please see our responses to Reviewer NKQs (Q4) for the parameter-matched baselines, and subband-specific temporal modeling, and Reviewer ULBe (Weakness) for the clarification regarding MSE metric.**


>Finally, the paper does not sufficiently position itself against recent wavelet, cascade, and spectral nowcasting methods, and the claims of broader applicability beyond precipitation forecasting are not supported.

**A.** We strengthened the positioning of DAWN-Cast by providing both conceptual and empirical comparisons with recent wavelet-, cascade-, and spectral-based nowcasting methods. Specifically, **response to Reviewer NKQs (Q1)** clarifies the conceptual differences between DAWN-Cast and prior wavelet, cascade, Fourier, and learned multi-resolution approaches, while **response to Reviewer NKQs (Q6)** complements this with quantitative comparisons, including predictive performance, parameter count, FLOPs, and inference throughput.

Regarding broader applicability, our intention was not to claim empirical validation beyond precipitation forecasting, but rather to motivate the generality of the proposed adaptive temporal modelling mechanism. The evidence supporting this claim is provided through the adaptive Gabor analyses in the paper. Table 3 shows that learnable Gabor activations consistently outperform fixed Gabor activations, while Table 4 demonstrates that different Gabor configurations are preferred under different precipitation regimes. In addition, **response to Reviewer NKQs (Q1)** presents a sinusoidal frequency sweep showing that the optimal Gabor configuration varies across datasets, indicating that the adaptive mechanism specializes to different dynamics rather than a single fixed operating regime.


### PDCR: 

>The recommendation could improve with controlled comparisons against recent wavelet and multi-scale nowcasting methods, together with parameter-matched ablations isolating wavelet decomposition, the Gabor stream, SRST, and FACL. Stronger evidence connecting latent subbands to interpretable radar structures would also be important.

We now included new controlled comparisons with recent wavelet and cascade-based nowcasting methods under the same evaluation protocol in **response to Reviewer NKQs Q6**, together with a detailed discussion of how DAWN-Cast differs conceptually from Fourier-, cascade-, multi-resolution-, and wavelet-based approaches in **response to Reviewer NKQs Q1**. To isolate the contribution of the proposed temporal modelling, we added parameter-matched shared and seprate scale specific temporal baseline ablation and progressive component-wise ablations (**response to Reviewer NKQs Q4**). Specific component wise ablations for isolating wavelet decomposition, the Gabor stream, SRST, and FACL are present in the paper in Table 2., to further strengthen the interpretation of the latent-space decomposition, we performed additional analyses directly on the latent representations and compared them with the original radar fields using spatial autocorrelation, low-frequency spectral energy, and power spectral density statistics, showing that the latent subbands preserve the large-scale/fine-scale distinctions observed in the original precipitation fields. We also clarified the spatial correspondence between the latent and pixel representations in our **response to Reviewer NKQs (Q2 and Q3)**.



>The authors should clarify why threshold-based metrics improve while MSE does not, and moderate claims regarding physical and climatic interpretability. Comparisons with Fourier decomposition, learned multi-resolution features, or shared temporal processing would further strengthen the paper.

We believe the rebuttal directly addresses these concerns. We clarified in our **response to Reviewer ULBe (Weakness)**  why threshold-based verification metrics (e.g., CSI/HSS) improve despite smaller changes in MSE by explaining the limitations of point-wise error for precipitation nowcasting and providing additional analyses showing that lower MSE does not necessarily correspond to improved threshold-based forecast skill.

Regarding the physical interpretation of the wavelet decomposition and the adaptive temporal modelling provided by the FAT blocks, we moderated our original claims and provided additional empirical evidence in **response to Reviewer NKQs (Q1)**. Specifically, we performed a sinusoidal frequency sweep spanning near-linear to oscillatory Gabor initializations, showing that different datasets consistently favor different frequency initializations. Together with the adaptive Gabor analyses presented in Tables 3 and 4, these results support the interpretation that the FAT blocks adapt their temporal behaviour to the statistical precipitation regime represented by the training data, rather than imposing a fixed temporal inductive bias.

For comparisons with Fourier decomposition and learned multi-resolution approaches, **response to Reviewer NKQs (Q1)** provides a detailed conceptual comparison with representative methods including FourCastNet, AlphaPre, EarthFormer, and SimVP. Regarding shared temporal processing, we added a parameter-matched shared-FAT baseline **(refer Reviewer bUgb Weakness 2 Table 2)** and further included the ablations reported in Table 2 of paper, where removing the wavelet decomposition or replacing the Gabor stream with a shared MLP demonstrates the importance of scale-specific temporal modelling.





