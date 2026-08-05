> Hi, thank you for your comment and taking the time to explain things. Unfortunately, my main point seems to not be addressed. I am not sure what you (the authors) mean by "better" I understand that MSE has many drawbacks when measuring meteorological variables as you pointed out. I also understand that different error measures will lead to different conclusions. In my opinion, especially if this is not a use case but a general contribution, better needs to be pinpointed and motivated in a falsifiable way. With enough error measures each method will win in one. It is therefore important to declare which falsifiable aspect you want to improve and which measure can measure this aspect. I do not see that you did that in either the paper or the answer.

> Without this, it seems like an idea: The idea is to separate scales. But the idea is not new enough to be novel in general and I can not see enough motivation understand how to falsify that the idea is good for this example.


**A.** Thank you for clarifying the criticism. We understand the distinction now: our previous response explained why we expect scale-specific temporal modelling to help, but did not sufficiently specify **what forecast deficiency we aim to improve and what observation would falsify that hypothesis. We do so now.**  


As we already discuss in the paper (Related Work and Sec. 3.2.2), models trained with pixel-wise losses tend to produce blurred predictions and can underestimate precipitation intensity at longer lead times. Our corresponding hypothesis is that explicitly modelling the temporal evolution of different wavelet subbands should better preserve precipitation structures, particularly at higher precipitation thresholds and longer lead times.

This hypothesis can be tested using CSI/HSS at predefined precipitation thresholds and across lead times. It would be unsupported if (i) the improvement did not persist or strengthen at higher precipitation thresholds, or (ii) the advantage disappeared at longer lead times.

**Test 1**: performance across precipitation intensity thresholds. Relative to the best baseline, the CSI improvements are:
| Dataset  |  CSI-M | Mid threshold | High threshold |
| -------- | -----: | ------------: | -------------: |
| SEVIR    | +11.6% |        +46.4% |         +85.1% |
| MeteoNet |  +6.8% |         +9.7% |         +10.6% |
| Shanghai |  +4.5% |         +4.0% |          +9.9% |
| CIKM     |  +3.4% |        +13.6% |          +9.2% |


The gain gets bigger as the rain gets heavier, on three of four datasets. A model that won by making smoother predictions would show the opposite. This is one metric family, sorted by the physical thing our method acts on. It is not picking a lucky metric out of many.


**Test 2**: performance across lead times. Figure 3 shows that on CIKM, DAWN-Cast maintains higher CSI and HSS relative to the compared baselines as lead time increases. If this advantage disappeared at longer lead times, it would not support our motivation concerning longer-horizon precipitation evolution.

**Test 3**: the MSE trade-off is systematic, not cherry-picked

You said that with enough metrics, every method wins one. Here is why that does not apply.

On SEVIR, five of our nine ablation rows (Refer Table 2 of paper) have better MSE than the full model. All five have worse high-threshold CSI:

| Ablation | MSE↓ | CSI-219↑ |
|:--|--:|--:|
| w/o Wavelet | 362.99 | 0.1056 |
| w/o WGTM | 360.20 | 0.0979 |
| w/o Channel-Mixing | 365.39 | 0.0717 |
| w/o Gabor | 366.26 | 0.0994 |
| w/o Spatial | 370.27 | 0.0997 |
| Ours | 371.34 | 0.1077 |

On MeteoNet, four of nine beat us on MSE (Refer Table 2 of paper). All four have worse CSI-32. There is no exception in either table.

This shows that the MSE/strong-precipitation trade-off is also present across controlled architectural ablations rather than only in the comparison with external baselines. We also discussed similar point on our precvious response. 


**Why MSE cannot test our claim?**

MSE is lowest when a model predicts a blurry average. If a model predicts a sharp storm core and puts it slightly in the wrong place, MSE punishes it twice: once for missing the real core, once for the false one. This is the known "double penalty" problem (Subich et al., ICML 2025; Roberts & Lean, 2008).

Concurrent work finds the same thing. WADEPre [1] reports worse RMSE than AlphaPre on SEVIR while improving CSI-219 by about 41%, and explains it with the same double penalty. So our MSE result is what our explanation predicts, not a gap we ignored.


**On novelty**, we agree that scale separation or wavelets alone are not novel, and we do not intend to claim otherwise. Our narrower methodological claim is that DAWN-Cast uses the wavelet decomposition to assign separate FAT temporal modelling to different subbands. With the wavelet decomposition retained, using the same FAT configuration across scales underperforms the scale-specific configuration:

Shared FAT (low-frequency configuration): SEVIR 0.3435, CIKM 0.3201
Shared FAT (high-frequency configuration): SEVIR 0.3444, CIKM 0.3219
Scale-specific FAT: SEVIR 0.3638, CIKM 0.3303

These controlled comparisons provide a direct test of whether scale-specific temporal treatment contributes beyond performing the wavelet decomposition itself.


References: [1] Liu, Baitian, et al. "WADEPre: A Wavelet-based Decomposition Model for Extreme Precipitation Nowcasting with Multi-Scale Learning." arXiv preprint arXiv:2602.02096 (2026).