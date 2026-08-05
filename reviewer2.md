Thank you for the clear criticism. You told us why you think DAWN-Cast should work. You never told us what result, if we saw it, would mean DAWN-Cast doesn't work. We do that now.

DAWN-Cast is built to fix one problem: models lose the strong rain cores at long lead times. The predictions get smooth and weak. (Strong claim weaken it, they might say why you did not add this in the paper)

The right way to measure this is CSI and HSS at each intensity threshold separately, plus frequency bias at high thresholds over lead time.


MSE is not the right measure for this, and we explain why below.

What would have proved us wrong

If our CSI gain stayed flat or got smaller as the rain threshold went up. That would mean we only improved easy, light-rain pixels.
If our CSI and HSS dropped with lead time at the same rate as the baselines. That would mean no long-range benefit.
If MSE improved everywhere at the same time as high-threshold CSI. That would mean our explanation (sharp vs. blurry) is not what is happening.

Test 1: gains grow with rain intensity

Improvement over the best baseline:

Dataset	CSI-M	Mid threshold	High threshold
SEVIR	+11.6%	+46.4%	+85.1%
MeteoNet	+6.8%	+9.7%	+10.6%
Shanghai	+4.5%	+4.0%	+9.9%
CIKM	+3.4%	+13.6%	+9.2%

The gain gets bigger as the rain gets heavier, on three of four datasets. A model that won by making smoother predictions would show the opposite. This is one metric family, sorted by the physical thing our method acts on. It is not picking a lucky metric out of many.

Test 3: the MSE trade-off is systematic, not cherry-picked

You said that with enough metrics, every method wins one. Here is why that does not apply.

On SEVIR, five of our nine ablation rows have better MSE than the full model. All five have worse high-threshold CSI:

w/o Wavelet: MSE 362.99, CSI-219 0.1056
w/o WGTM: 360.20, 0.0979
w/o Channel-Mixing: 365.39, 0.0717
w/o Gabor: 366.26, 0.0994
w/o Spatial: 370.27, 0.0997
Ours: 371.34, 0.1077

On MeteoNet, four of nine beat us on MSE. All four have worse CSI-32. There is no exception in either table.

Nine different changes to the model all push MSE and heavy-rain CSI in opposite directions. This is a stable property of the architecture, not a metric we picked afterwards.

Why MSE cannot test our claim

MSE is lowest when a model predicts a blurry average. If a model predicts a sharp storm core and puts it slightly in the wrong place, MSE punishes it twice: once for missing the real core, once for the false one. This is the known "double penalty" problem (Subich et al., ICML 2025; Roberts & Lean, 2008).

Concurrent work finds the same thing. WADEPre (arXiv:2602.02096) reports worse RMSE than AlphaPre on SEVIR while improving CSI-219 by about 41%, and explains it with the same double penalty. So our MSE result is what our explanation predicts, not a gap we ignored.

(We expect the reviewer to give us more time to do experiments. He replied last moment)

Both can fail clearly. If our frequency bias falls off like the baselines, or our high-wavenumber energy is no closer to the truth, our claim is wrong.


On novelty: 

We agree that "use a wavelet transform" is not novel, and we withdraw any such claim. (dont say we agree)

Our actual claim is narrower: giving each subband its own separately-tuned temporal operator is what produces the gain. Our test, with the wavelet transform present in all three settings:

Shared FAT (low-freq settings): SEVIR CSI-M 0.3435, CIKM 0.3201
Shared FAT (high-freq settings): SEVIR 0.3444, CIKM 0.3219
Separate FAT per band: SEVIR 0.3638, CIKM 0.3303


------------------------------------------------------

Thank you for the clear criticism. You told us why we think DAWN-Cast should work. You never told us what result would mean DAWN-Cast doesn't work. We do that now.

As we already state in the paper (related work, and Section 3.2.2), models trained with pixel-wise loss tend to blur their predictions, which weakens strong rain cores at longer lead times. DAWN-Cast is built to fix this specific weakness.

The right way to measure this is CSI and HSS at each rain intensity threshold separately, plus frequency bias at high thresholds over lead time. MSE is not the right measure for this, and we explain why below.

**What would have proved us wrong**

1. If our CSI gain stayed flat or got smaller as the rain threshold went up. That would mean we only improved easy, light-rain pixels.
2. If our CSI and HSS dropped with lead time at the same rate as the baselines. That would mean no long-range benefit.
3. If MSE improved everywhere at the same time as high-threshold CSI. That would mean our explanation (sharp vs. blurry) is not what is happening.

**Test 1: gains grow with rain intensity**

Improvement over the best baseline:

| Dataset | CSI-M | Mid threshold | High threshold |
|---|---|---|---|
| SEVIR | +11.6% | +46.4% | +85.1% |
| MeteoNet | +6.8% | +9.7% | +10.6% |
| Shanghai | +4.5% | +4.0% | +9.9% |
| CIKM | +3.4% | +13.6% | +9.2% |

The gain gets bigger as the rain gets heavier, on three of four datasets. A model that won by making smoother predictions would show the opposite. This is one metric family, sorted by the physical thing our method acts on — not a lucky metric picked out of many.

**Test 2: the advantage holds at long lead times**

As shown in Figure 3, DAWN-Cast's CSI and HSS decay more slowly with lead time than every baseline on CIKM. This is the second falsifier: if our advantage disappeared at later lead times, it would mean we only help early in the forecast, when the task is easy. It does not disappear.

**Test 3: the MSE trade-off is systematic, not cherry-picked**

You said that with enough metrics, every method wins one. Here is why that does not apply here.

On SEVIR, five of our nine ablation rows have better MSE than the full model. All five have worse high-threshold CSI:

- w/o Wavelet: MSE 362.99, CSI-219 0.1056
- w/o WGTM: 360.20, 0.0979
- w/o Channel-Mixing: 365.39, 0.0717
- w/o Gabor: 366.26, 0.0994
- w/o Spatial: 370.27, 0.0997
- Ours: 371.34, 0.1077

On MeteoNet, four of nine beat us on MSE. All four have worse CSI-32. There is no exception in either table.

Nine different changes to the model all push MSE and heavy-rain CSI in opposite directions. This is a stable property of the architecture, not a metric we picked afterwards.

**Why MSE cannot test our claim**

MSE is lowest when a model predicts a blurry average. If a model predicts a sharp storm core and puts it slightly in the wrong place, MSE punishes it twice: once for missing the real core, once for the false one. This is the known "double penalty" problem (Subich et al., ICML 2025; Roberts & Lean, 2008).

Concurrent work finds the same pattern. WADEPre (arXiv:2602.02096) reports worse RMSE than AlphaPre on SEVIR while improving CSI-219 by about 41%, and explains it with the same double penalty. So our MSE result is what our explanation predicts, not a gap we are ignoring.

**Measurements we will add in the revision**

To test the aspect directly, instead of through a stand-in metric, we will add:

1. Frequency bias at high thresholds, per lead time — a direct measure of whether the model under- or over-predicts heavy rain later in the forecast.
2. Radial power spectrum of our predictions vs. ground truth at high wavenumbers, at the last lead time. Appendix C already does this for the wavelet bands of the ground truth; we will extend it to model outputs.

Both can fail clearly. If our frequency bias falls off like the baselines, or our high-wavenumber energy is no closer to the truth, our claim is wrong.

**On novelty**

To be precise about what we are and are not claiming here: using a wavelet transform by itself is well known and we are not claiming otherwise.

Our actual claim is narrower: giving each subband its own separately-tuned temporal operator is what produces the gain. Our test, with the wavelet transform present in all three settings:

- Shared FAT (low-freq settings): SEVIR CSI-M 0.3435, CIKM 0.3201
- Shared FAT (high-freq settings): SEVIR 0.3444, CIKM 0.3219
- Separate FAT per band: SEVIR 0.3638, CIKM 0.3303

We should also correct something from our earlier response. We described WADEPre as using the same temporal treatment across subbands. That is not accurate — WADEPre uses two separate networks, one for the low-frequency band (A-Net) and one for the high-frequency band (D-Net). The real difference from our approach is that our temporal operator's frequency behaviour is learned per subband, rather than set by using two fixed, differently-built networks, and that we operate in a compressed latent space rather than pixel space. We will correct our related-work discussion to reflect this.

Separately, we also noticed an error in the paper: line 232 states we achieve the lowest SEVIR MSE, which is incorrect (AlphaPre is lower). We will fix this in the revision.