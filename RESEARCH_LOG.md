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
**Next:** Train amplinet only on mse, in latent space and also on amplitude difference seperately. 

## 2026-02-02 | Ran Amplinet with only fal-fcl loss
**Goal:** to Check how it is better than mse loss.
**Change:** Removed mse and added fal-fcl loss. 
**Why:** 
**Result:** 
**Next:** 