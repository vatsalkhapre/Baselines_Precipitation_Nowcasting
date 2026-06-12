#!/bin/bash

GPU_ID=1
RUNNER=run_alphapre_convlstm_sevir_lr_latent.py
AE_CKPT_DIR=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints

# =============================================================================
# LPCast Ablations - SEVIR
# =============================================================================
# Ablation 1: LPCast_onlyspatial
# Ablation 2: LPCast_onlytemporal
# Ablation 3: LPCast_wspatiotemporal
# =============================================================================


# =============================================================================
# Ablation 1: LPCast_onlyspatial
# =============================================================================

# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast_onlyspatial \
#     --dataset           sevir_lr_latent_32 \
#     --img_channel       4 \
#     --img_size          32 \
#     --frames_in         5 \
#     --frames_out        20 \
#     --seq_len           25 \
#     --hidden_dim        64 \
#     --mlp_size_factor   1.0 \
#     --lift_dims         64 64 64 \
#     --proj_dims         64 64 4 \
#     --facl_const_ratio  0.1 \
#     --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth \
#     --epochs            50 \
#     --batch_size        4 \
#     --num_workers       8 \
#     --exp_dir           lpcast_ablations_sevir \
#     --exp_note          sevir_onlyspatial_dim64 \
#     --wandb_state       offline \
#     --wandb_project_name ACML \
#     --run_name          LPCast_onlyspatial_SEVIR \
#     --gpu_use           $GPU_ID \
#     --valid

# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast_onlyspatial \
#     --dataset           sevir_lr_latent_32 \
#     --img_channel       4 \
#     --img_size          32 \
#     --frames_in         5 \
#     --frames_out        20 \
#     --seq_len           25 \
#     --hidden_dim        64 \
#     --mlp_size_factor   1.0 \
#     --lift_dims         64 64 64 \
#     --proj_dims         64 64 4 \
#     --facl_const_ratio  0.1 \
#     --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth \
#     --epochs            50 \
#     --batch_size        4 \
#     --num_workers       8 \
#     --exp_dir           lpcast_ablations_sevir \
#     --exp_note          sevir_onlyspatial_dim64 \
#     --wandb_state       offline \
#     --gpu_use           $GPU_ID \
#     --eval


# =============================================================================
# Ablation 2: LPCast_onlytemporal
# =============================================================================

CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
    --backbone          LPCast_onlytemporal \
    --dataset           sevir_lr_latent_32 \
    --img_channel       4 \
    --img_size          32 \
    --frames_in         5 \
    --frames_out        20 \
    --seq_len           25 \
    --hidden_dim        64 \
    --mlp_size_factor   1.0 \
    --lift_dims         64 64 64 \
    --proj_dims         64 64 4 \
    --facl_const_ratio  0.1 \
    --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth \
    --epochs            50 \
    --batch_size        4 \
    --num_workers       8 \
    --exp_dir           lpcast_ablations_sevir \
    --exp_note          sevir_onlytemporal_dim64 \
    --wandb_state       offline \
    --wandb_project_name ACML \
    --run_name          LPCast_onlytemporal_SEVIR \
    --gpu_use           $GPU_ID \
    --valid

# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast_onlytemporal \
#     --dataset           sevir_lr_latent_32 \
#     --img_channel       4 \
#     --img_size          32 \
#     --frames_in         5 \
#     --frames_out        20 \
#     --seq_len           25 \
#     --hidden_dim        64 \
#     --mlp_size_factor   1.0 \
#     --lift_dims         64 64 64 \
#     --proj_dims         64 64 4 \
#     --facl_const_ratio  0.1 \
#     --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth \
#     --epochs            50 \
#     --batch_size        4 \
#     --num_workers       8 \
#     --exp_dir           lpcast_ablations_sevir \
#     --exp_note          sevir_onlytemporal_dim64 \
#     --wandb_state       offline \
#     --gpu_use           $GPU_ID \
#     --eval


# =============================================================================
# Ablation 3: LPCast_wspatiotemporal
# =============================================================================

# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast_wospatiotemporal \
#     --dataset           sevir_lr_latent_32 \
#     --img_channel       4 \
#     --img_size          32 \
#     --frames_in         5 \
#     --frames_out        20 \
#     --seq_len           25 \
#     --hidden_dim        64 \
#     --mlp_size_factor   1.0 \
#     --lift_dims         64 64 64 \
#     --proj_dims         64 64 4 \
#     --facl_const_ratio  0.1 \
#     --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth \
#     --epochs            50 \
#     --batch_size        4 \
#     --num_workers       8 \
#     --exp_dir           lpcast_ablations_sevir \
#     --exp_note          sevir_wspatiotemporal_dim64 \
#     --wandb_state       offline \
#     --wandb_project_name ACML \
#     --run_name          LPCast_wspatiotemporal_SEVIR \
#     --gpu_use           $GPU_ID \
#     --valid

# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast_wspatiotemporal \
#     --dataset           sevir_lr_latent_32 \
#     --img_channel       4 \
#     --img_size          32 \
#     --frames_in         5 \
#     --frames_out        20 \
#     --seq_len           25 \
#     --hidden_dim        64 \
#     --mlp_size_factor   1.0 \
#     --lift_dims         64 64 64 \
#     --proj_dims         64 64 4 \
#     --facl_const_ratio  0.1 \
#     --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth \
#     --epochs            50 \
#     --batch_size        4 \
#     --num_workers       8 \
#     --exp_dir           lpcast_ablations_sevir \
#     --exp_note          sevir_wspatiotemporal_dim64 \
#     --wandb_state       offline \
#     --gpu_use           $GPU_ID \
#     --eval