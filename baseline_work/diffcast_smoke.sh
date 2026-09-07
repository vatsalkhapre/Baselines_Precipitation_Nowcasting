#!/bin/bash
# Mandatory gate 1 for DiffCast+FACL: short CIKM run to prove the FACL backbone
# loss and the native diffusion residual loss combine and train end to end.
GPU=${1:-0}
R=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting
PY=/home/vatsal/miniconda3/envs/earthformer/bin/python
rm -rf $R/Exps/diffsmoke
CUDA_VISIBLE_DEVICES=$GPU $PY $R/run_diffcast_falfcl.py \
  --exp_dir diffsmoke --exp_note diffcast_falfcl_smoke \
  --backbone phydnet_falfcl --use_diff --dataset cikm \
  --batch_size 4 --seq_len 25 --frames_in 5 --frames_out 10 --img_size 128 \
  --epochs 2 --valid --valid_limit --vlnum 2 \
  --num_workers 4 --wandb_state offline
