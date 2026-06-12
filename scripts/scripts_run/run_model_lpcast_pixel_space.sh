#!/bin/bash
# =============================================================================
# LPCast — SEVIR Best Configuration (Pixel Space)
# Runner  : run_alphapre_convlstm.py
# Dataset : SEVIR  |  dim=64  |  best epoch=66
# Edit GPU_ID before running.
# =============================================================================

set -e   # abort on first error

GPU_ID=2
RUNNER="run_alphapre_convlstm.py"

echo "========== LPCast SEVIR (pixel space) — dim=64 =========="
CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
    --backbone          lpcast_mse \
    --dataset           sevir \
    --img_channel       1 \
    --img_size          128 \
    --frames_in         5 \
    --frames_out        20 \
    --seq_len           25 \
    --hidden_dim        64 \
    --mlp_size_factor   1.0 \
    --facl_const_ratio  0.1 \
    --lift_dims         64 64 64 \
    --proj_dims         64 64 1 \
    --epochs            50 \
    --batch_size        4 \
    --num_workers       8 \
    --exp_dir           lpcast_mse_sevir_pixel \
    --exp_note          sevir_dim64 \
    --run_name          LPCast_mse_SEVIR_pixel\
    --wandb_state        online \
    --wandb_project_name  ACML \
    --gpu_use           $GPU_ID \
    --valid
echo "========== SEVIR done =========="