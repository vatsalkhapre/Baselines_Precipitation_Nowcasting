## 2026-01-16 | Exp-014
**Goal:** Improve short-horizon rainfall skill
**Change:** Switched loss from L2 → Lp (p=1.5), added orography channel
**Why:** L2 over-smoothing extreme cells
**Result:** +6% SSIM, unstable training after epoch 18
**Next:** Try gradient clipping + lower LR