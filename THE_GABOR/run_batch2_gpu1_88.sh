#!/bin/bash
# Batch 2 -- runs SEQUENTIALLY on GPU 1 of this machine (10.24.52.88).
# All load the SAME shared DAWN-Cast init (sha 05e5ab4a...) as Exps 1-3.
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1
export WANDB_API_KEY="${WANDB_API_KEY:?set WANDB_API_KEY before running}"
mkdir -p THE_GABOR/logs/_runlogs
COMMON="--seed 0 --epochs 50 --val_every_epochs 5 --limit_train_batches 2000 \
--limit_val_batches 200 --batch_size 4 --num_workers 8 --lr 1e-4 \
--wave db6 --wavelet_level 2 --hf_mode separate --hidden_dim 64 \
--afno_blocks 4 --afno_hidden_size_factor 4 --sparsity_threshold 0.01 --k_spatial 3 \
--ae_ckpt_path Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth \
--wandb_project THE_GABOR --wandb_state online"

# Exp 4: target RANDOM, Gabor from RANDOM donor (matched)
N=DAWNCast_latent_random_gaborinit_random_seed0
echo "=== $N ==="; python -m THE_GABOR.run_dawncast_transfer $COMMON \
  --target_regime random --donor_regime random --transfer gabor \
  --run_name $N > THE_GABOR/logs/_runlogs/$N.log 2>&1

# Exp 5: target RANDOM, Gabor from STORM donor (mismatched)
N=DAWNCast_latent_random_gaborinit_storm_seed0
echo "=== $N ==="; python -m THE_GABOR.run_dawncast_transfer $COMMON \
  --target_regime random --donor_regime storm --transfer gabor \
  --run_name $N > THE_GABOR/logs/_runlogs/$N.log 2>&1

# Exp 7: target STORM, same transfer as Exp 3 but ONLY the Gabor is frozen
#         (MLP stays trainable) -> isolates the freeze scope against Exp 3.
N=DAWNCast_latent_storm_freezegabor_storm_seed0
echo "=== $N ==="; python -m THE_GABOR.run_dawncast_transfer $COMMON \
  --target_regime storm --donor_regime storm \
  --transfer gabor mlp lifting projection --freeze gabor \
  --run_name $N > THE_GABOR/logs/_runlogs/$N.log 2>&1

echo "batch 2 complete"
