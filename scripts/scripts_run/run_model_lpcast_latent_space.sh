#!/bin/bash
# =============================================================================
# LPCast — Best Configuration Sweep (Latent Space)
# Runner  : run_alphapre_convlstm_sevir_lr_latent.py
# Datasets: SEVIR → Shanghai → Meteonet → CIKM  (sequential)
# Edit GPU_ID and AE_CKPT_DIR before running.
# =============================================================================

set -e   # abort on first error

GPU_ID=1
AE_CKPT_DIR="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints"
RUNNER="run_alphapre_convlstm_sevir_lr_latent.py"

# =============================================================================
# 1. SEVIR  |  dim=64  |  best epoch=66
# =============================================================================
# echo "========== [1/4] SEVIR — dim=64 =========="
# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast \
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
#     --exp_dir           lpcast_latent_best \
#     --exp_note          sevir_dim64 \
#     --run_name          LPCast_SEVIR_dim64 \
#     --wandb_state        online \
#     --wandb_project_name  ACML \
#     --run_name           LPCast_sevir \
#     --gpu_use           $GPU_ID \
#     --valid
# echo "========== [1/4] SEVIR — dim=64 =========="
# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast \
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
#     --exp_dir           lpcast_latent_best \
#     --exp_note          sevir_dim64 \
#     --run_name          LPCast_SEVIR_dim64 \
#     --wandb_state        online \
#     --wandb_project_name  ACML \
#     --run_name           LPCast_sevir \
#     --gpu_use           $GPU_ID \
#     --valid

# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast \
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
#     --exp_dir           lpcast_latent_best \
#     --exp_note          sevir_dim64 \
#     --run_name          LPCast_SEVIR_dim64 \
#     --wandb_state        offline \
#     --gpu_use           $GPU_ID \
#     --eval
# echo "========== SEVIR done =========="
# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast \
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
#     --exp_dir           lpcast_latent_best \
#     --exp_note          sevir_dim64 \
#     --run_name          LPCast_SEVIR_dim64 \
#     --wandb_state        offline \
#     --gpu_use           $GPU_ID \
#     --eval
# echo "========== SEVIR done =========="

# =============================================================================
# 2. Shanghai  |  dim=64  |  best epoch not recorded — using 50
# =============================================================================
echo "========== [2/4] Shanghai — dim=64 =========="
CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
    --backbone          LPCast \
    --dataset           shanghai_lr_latent_32 \
    --img_channel       4 \
    --img_size          32 \
    --frames_in         5 \
    --frames_out        20 \
    --seq_len           25 \
    --hidden_dim        64 \
    --mlp_size_factor   1.0 \
    --lift_dims         64 64 64 \
    --proj_dims         64 64 64 4 \
    --facl_const_ratio  0.1 \
    --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SHANGHAI.pth \
    --epochs            50 \
    --batch_size        4 \
    --num_workers       8 \
    --exp_dir           lpcast_latent_best \
    --exp_note          shanghai_dim64 \
    --run_name          LPCast_Shanghai_dim64 \
    --wandb_state        offline \
    --wandb_project_name  ACML \
    --run_name           LPCast_shanghai \
    --gpu_use           $GPU_ID \
    --valid

CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
    --backbone          LPCast \
    --dataset           shanghai_lr_latent_32 \
    --img_channel       4 \
    --img_size          32 \
    --frames_in         5 \
    --frames_out        20 \
    --seq_len           25 \
    --hidden_dim        64 \
    --mlp_size_factor   1.0 \
    --lift_dims         64 64 64 \
    --proj_dims         64 64 64 4 \
    --facl_const_ratio  0.1 \
    --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_SHANGHAI.pth \
    --epochs            50 \
    --batch_size        4 \
    --num_workers       8 \
    --exp_dir           lpcast_latent_best \
    --exp_note          shanghai_dim64 \
    --run_name          LPCast_Shanghai_dim64 \
    --wandb_state        offline \
    --gpu_use           $GPU_ID \
    --eval
echo "========== Shanghai done =========="

# # =============================================================================
# # 3. Meteonet  |  dim=64  |  best epoch=70
# # =============================================================================
# echo "========== [3/4] Meteonet — dim=64 =========="
# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast \
#     --dataset           meteo_lr_latent_32 \
#     --img_channel       4 \
#     --img_size          32 \
#     --frames_in         5 \
#     --frames_out        20 \
#     --seq_len           25 \
#     --hidden_dim        64 \
#     --mlp_size_factor   1.0 \
#     --lift_dims         64 64 64 \
#     --proj_dims         64 64 64 4 \
#     --facl_const_ratio  0.1 \
#     --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_METEONET.pth \
#     --epochs            50 \
#     --batch_size        4 \
#     --num_workers       8 \
#     --exp_dir           lpcast_latent_best \
#     --exp_note          meteonet_dim64 \
#     --run_name          LPCast_Meteonet_dim64 \
#     --wandb_state        online \
#     --wandb_project_name  ACML \
#     --run_name           LPCast_meteonet \
#     --gpu_use           $GPU_ID \
#     --valid

# CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#     --backbone          LPCast \
#     --dataset           meteo_lr_latent_32 \
#     --img_channel       4 \
#     --img_size          32 \
#     --frames_in         5 \
#     --frames_out        20 \
#     --seq_len           25 \
#     --hidden_dim        64 \
#     --mlp_size_factor   1.0 \
#     --lift_dims         64 64 64 \
#     --proj_dims         64 64 64 4 \
#     --facl_const_ratio  0.1 \
#     --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_METEONET.pth \
#     --epochs            50 \
#     --batch_size        4 \
#     --num_workers       8 \
#     --exp_dir           lpcast_latent_best \
#     --exp_note          meteonet_dim64 \
#     --run_name          LPCast_Meteonet_dim64 \
#     --wandb_state        offline \
#     --gpu_use           $GPU_ID \
#     --eval
# echo "========== Meteonet done =========="

# =============================================================================
# 4. CIKM  |  dim=128  |  frames_out=10  |  best epoch=205
#    seq_len set to 15 (frames_in=5 + frames_out=10)
# =============================================================================
echo "========== [4/4] CIKM — dim=128 =========="
CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
    --backbone          LPCast \
    --dataset           cikm_latent_32 \
    --img_channel       4 \
    --img_size          32 \
    --frames_in         5 \
    --frames_out        10 \
    --seq_len           15 \
    --hidden_dim        128 \
    --mlp_size_factor   1.0 \
    --lift_dims         128 128 128 \
    --proj_dims         128 128 128 4 \
    --facl_const_ratio  0.1 \
    --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_CIKM.pth \
    --epochs            50 \
    --batch_size        4 \
    --num_workers       8 \
    --exp_dir           lpcast_latent_best \
    --exp_note          cikm_dim128 \
    --run_name          LPCast_CIKM_dim128 \
    --wandb_state        offline \
    --wandb_project_name  ACML \
    --run_name           LPCast_cikm \
    --gpu_use           $GPU_ID \
    --valid

CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
    --backbone          LPCast \
    --dataset           cikm_latent_32 \
    --img_channel       4 \
    --img_size          32 \
    --frames_in         5 \
    --frames_out        10 \
    --seq_len           15 \
    --hidden_dim        128 \
    --mlp_size_factor   1.0 \
    --lift_dims         128 128 128 \
    --proj_dims         128 128 128 4 \
    --facl_const_ratio  0.1 \
    --ae_ckpt_path      $AE_CKPT_DIR/autoencoder_checkpoint_32_CIKM.pth \
    --epochs            50 \
    --batch_size        4 \
    --num_workers       8 \
    --exp_dir           lpcast_latent_best \
    --exp_note          cikm_dim128 \
    --run_name          LPCast_CIKM_dim128 \
    --wandb_state        offline \
    --gpu_use           $GPU_ID \
    --eval
# echo "========== CIKM done =========="

# echo "========== All 4 runs complete =========="