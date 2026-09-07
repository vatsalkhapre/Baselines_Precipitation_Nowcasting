#!/bin/bash
# DiffCast + PhyDNet backbone + FACL.
#
# LOSS PLACEMENT (the point of this whole variant):
#   models/diffcast.py::_predict computes
#       loss = 0.5 * diff_loss + 0.5 * backbone_loss
#   diff_loss  = p_losses on the RESIDUAL, a noise-prediction term -> UNCHANGED, native.
#   backbone_loss = the deterministic PhyDNet backbone's forecast loss -> FACL here.
#   FACL is never applied to noise, and no x0 is reconstructed from noise to force a
#   spatial loss onto the diffusion branch. The published diffusion procedure is intact.
#
#   usage: launch_diffcast_falfcl.sh <dataset> <gpu> [epochs]
set -u
DS=$1; GPU=$2
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
case $DS in
  cikm)     FO=10; EP=${3:-40} ;;
  shanghai) FO=20; EP=${3:-40} ;;
  meteo)    FO=20; EP=${3:-40} ;;
  # SEVIR is 5391 s/epoch (1.5 h). 15 epochs is what fits in a 24 h window.
  # Owner chose option (a): run what fits, report it, disclose as undertrained.
  sevir)    FO=20; EP=${3:-15} ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
exec env CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_diffcast_falfcl.py \
  --exp_dir diffcast_falfcl --exp_note diffcast_falfcl_on_$DS \
  --backbone phydnet_falfcl --use_diff --dataset $DS \
  --batch_size 4 --seq_len 25 --frames_in 5 --frames_out $FO --img_size 128 \
  --lr 1e-4 --epochs $EP --valid --num_workers 8 \
  --wandb_state online --wandb_project_name ICLR26_FACL_runs \
  --run_name diffcast_falfcl_$DS
