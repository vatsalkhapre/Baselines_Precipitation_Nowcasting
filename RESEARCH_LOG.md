## 2026-01-20 | Reduce alphapre layers
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
**Next:** 