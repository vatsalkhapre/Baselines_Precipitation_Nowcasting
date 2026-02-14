## 2026-01-19 | Changed the model alphapre latent in the latent file
**Goal:** To use correct model for latent
**Change:** switched latent model name
**Why:** So that sigmoid doesnt act on -1,1
**Result:** Results similar to training of origninal alphapre, but with lesser training time. 
**Next:** 

## 2026-01-21 | Ran Amplinet from alphapre
**Goal:** to Check amplinet effectiveness
**Change:** Extract amplinet from alphapre
**Why:** To check contribution of amplinet on alphapre
**Result:** It is good, I mean major portion of alphapre
**Next:** Train amplinet only on mse, in latent space and also on amplitude difference seperately. ## 2026-01-20 | Reduce alphapre layers
**Goal:** Training in latent space to reduce the training time, reducing layers to avoid overfitting.
**Change:** Layers 3 -> Layers 2
**Why:** To avoid overfitting. 
**Result:**  Performance remains same
**Next:** 

## 2026-01-20 | Resuming training with Original alphapre latent (layers 3)
**Goal:** To improve the score of best model
**Change:** lr 1e-4 -> 1e-5
**Why:** To improve performance
**Result:** 
**Next:** 

## 2026-01-27 | Here we have added afno for spatial refining.
**Goal:** To improve the spatial refining through afno
**Change:** replaced the alplitimecell with afnoamplitimecell, afno is placed after the timemlp so that it can use the processed time output and not create one. 
**Why:** For spatial refinement
**Result:** 
**Next:** ## 2026-01-19 | Changed the model alphapre latent in the latent file

**Goal:** To use correct model for latent
**Change:** switched latent model name
**Why:** So that sigmoid doesnt act on -1,1
**Result:** Results similar to training of origninal alphapre, but with lesser training time. 
**Next:** 

## 2026-01-21 | Ran Amplinet from alphapre
**Goal:** to Check amplinet effectiveness
**Change:** Extract amplinet from alphapre
**Why:** To check contribution of amplinet on alphapre
**Result:** It is good, I mean major portion of alphapre
**Next:** Train amplinet only on mse, in latent space and also on amplitude difference seperately. 

## 2026-02-02 | Ran Amplinet with only fal-fcl loss
**Goal:** to Check how it is better than mse loss.
**Change:** Removed mse and added fal-fcl loss. 
**Why:** 
**Result:** 
**Next:** 

## 2026-02-04 | Here we have created a new variant for fno.
**Goal:** To improve the spatial refining through afno
**Change:** Had some small changes have a seperate file for (alphapre_fnoamplinet_MSE_only_another_variant.py). 
**Why:** 
**Result:** 
**Next:** To see if the results improve

## 2026-02-08 | Made pipelines for wavelet LL, wavelet High and wavelet full
**Goal:**  Learn precipitation nowcasting in wavelet space. 
**Change:**  Whole new 3 pipelines
**Why:** Novelity
**Result:** okisshhhhh
**Next:** Change model and try.