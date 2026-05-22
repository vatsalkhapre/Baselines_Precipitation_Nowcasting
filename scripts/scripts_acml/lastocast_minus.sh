#!/bin/bash
# =============================================================================
# Run: alpha_amplinet_latent_FAL_FCL_2_3_13_2_mse across all datasets
# Backbone resolved via registry: amplinet_latent_falfcl_only_2.3.13.2.mse
# kwargs_type: standared  (no gabor params)
# =============================================================================

set -e  # exit on any error

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
RUN_FILE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent_for_model_parts.py"
AE_CKPT_BASE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints"

# -----------------------------------------------------------------------------
# Fixed config
# -----------------------------------------------------------------------------
BACKBONE1="amplinet_latent_falfcl_only_2.3.13.2.BC"
BACKBONE2="amplinet_latent_falfcl_only_2.3.13.2.mse.BC"
BACKBONE3="amplinet_latent_falfcl_only_2.3.13.2.mse.BC"
IMG_SIZE=32
IMG_CHANNEL=4
EXP_DIR="2_3_13_2_mse_BC_runs"
EPOCHS=50
BATCH_SIZE=4
GRAD_ACC=8
LR=1e-4
WARMUP=1000
SCHEDULER="cosine"
WANDB_PROJECT="Alphapre"
WANDB_STATE="online"

# -----------------------------------------------------------------------------
# Dataset-specific AE checkpoints — fill these in if they differ
# -----------------------------------------------------------------------------
AE_CKPT_SEVIR="${AE_CKPT_BASE}/autoencoder_checkpoint_32_SEVIR.pth"
AE_CKPT_SHANGHAI="${AE_CKPT_BASE}/autoencoder_checkpoint_32_SHANGHAI.pth"
AE_CKPT_METEO="${AE_CKPT_BASE}/autoencoder_checkpoint_32_METEONET.pth"
AE_CKPT_CIKM="${AE_CKPT_BASE}/autoencoder_checkpoint_32_CIKM.pth"


# =============================================================================
# 1. SEVIR  (frames_in=5, frames_out=20)
# =============================================================================
# echo "============================================================"
# echo " Running: sevir_lr_latent_32"
# echo "============================================================"
# python3 "${RUN_FILE}" \
#     --backbone          "${BACKBONE}" \
#     --dataset           sevir_lr_latent_32 \
#     --img_size          ${IMG_SIZE} \
#     --gpu_use           1 \
#     --img_channel       ${IMG_CHANNEL} \
#     --frames_in         5 \
#     --frames_out        20 \
#     --exp_dir           "${EXP_DIR}" \
#     --exp_note          sevir_lr_latent_32 \
#     --epochs            ${EPOCHS} \
#     --batch_size        ${BATCH_SIZE} \
#     --wandb_state       ${WANDB_STATE} \
#     --wandb_project_name "${WANDB_PROJECT}" \
#     --run_name          "lastocast_minus_mse_sevir" \
#     --ae_ckpt_path      "${AE_CKPT_SEVIR}" \
#     --valid \
#     --seed 0


# =============================================================================
# 2. Shanghai  (frames_in=5, frames_out=20)
# =============================================================================
# echo "============================================================"
# echo " Running: shanghai_lr_latent_32"
# echo "============================================================"
# python3 "${RUN_FILE}" \
#     --backbone          "${BACKBONE}" \
#     --dataset           shanghai_lr_latent_32 \
#     --img_size          ${IMG_SIZE} \
#     --gpu_use           1 \
#     --img_channel       ${IMG_CHANNEL} \
#     --frames_in         5 \
#     --frames_out        20 \
#     --exp_dir           "${EXP_DIR}" \
#     --exp_note          shanghai_lr_latent_32 \
#     --epochs            ${EPOCHS} \
#     --batch_size        ${BATCH_SIZE} \
#     --wandb_state       ${WANDB_STATE} \
#     --wandb_project_name "${WANDB_PROJECT}" \
#     --run_name          "lastocast_minus_mse_shanghai" \
#     --ae_ckpt_path      "${AE_CKPT_SHANGHAI}" \
#     --valid \
#     --seed 0


# =============================================================================
# 3. MeteoNet  (frames_in=5, frames_out=20)
# =============================================================================
echo "============================================================"
echo " Running: meteo_lr_latent_32"
echo "============================================================"
python3 "${RUN_FILE}" \
    --backbone          "${BACKBONE}" \
    --dataset           meteo_lr_latent_32 \
    --img_size          ${IMG_SIZE} \
    --gpu_use           1 \
    --img_channel       ${IMG_CHANNEL} \
    --frames_in         5 \
    --frames_out        20 \
    --exp_dir           "${EXP_DIR}" \
    --exp_note          meteo_lr_latent_32 \
    --epochs            ${EPOCHS} \
    --batch_size        ${BATCH_SIZE} \
    --wandb_state       ${WANDB_STATE} \
    --wandb_project_name "${WANDB_PROJECT}" \
    --run_name          "lastocast_minus_mse_meteo" \
    --ae_ckpt_path      "${AE_CKPT_METEO}" \
    --valid \
    --seed 0


echo "============================================================"
echo " Running: meteo_lr_latent_32_EVAL"
echo "============================================================"
python3 "${RUN_FILE}" \
    --backbone          "${BACKBONE}" \
    --dataset           meteo_lr_latent_32 \
    --img_size          ${IMG_SIZE} \
    --gpu_use           1 \
    --img_channel       ${IMG_CHANNEL} \
    --frames_in         5 \
    --frames_out        20 \
    --exp_dir           "${EXP_DIR}" \
    --exp_note          meteo_lr_latent_32 \
    --epochs            ${EPOCHS} \
    --batch_size        ${BATCH_SIZE} \
    --wandb_state       ${WANDB_STATE} \
    --wandb_project_name "${WANDB_PROJECT}" \
    --run_name          "lastocast_minus_mse_meteo" \
    --ae_ckpt_path      "${AE_CKPT_METEO}" \
    --eval \
    --seed 0

# =============================================================================
# 4. CIKM  (frames_in=5, frames_out=10)
# =============================================================================
# echo "============================================================"
# echo " Running: cikm_latent_32"
# echo "============================================================"
# python3 "${RUN_FILE}" \
#     --backbone          "${BACKBONE}" \
#     --dataset           cikm_latent_32 \
#     --img_size          ${IMG_SIZE} \
#     --gpu_use           1 \
#     --img_channel       ${IMG_CHANNEL} \
#     --frames_in         5 \
#     --frames_out        10 \
#     --exp_dir           "${EXP_DIR}" \
#     --exp_note          cikm_latent_32 \
#     --epochs            ${EPOCHS} \
#     --batch_size        ${BATCH_SIZE} \
#     --wandb_state       ${WANDB_STATE} \
#     --wandb_project_name "${WANDB_PROJECT}" \
#     --run_name          "lastocast_minus_mse_cikm" \
#     --ae_ckpt_path      "${AE_CKPT_CIKM}" \
#     --valid \
#     --seed 0


# echo "============================================================"
# echo " All dataset runs complete."
# echo "============================================================"