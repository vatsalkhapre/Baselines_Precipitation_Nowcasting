#!/bin/bash
# ==============================================================
# DAWNCast — Pixel Space Training
# Runner  : run_alphapre_convlstm.py
# Backbone: dawncast
# Space   : pixel  (raw frames, no autoencoder)
# ==============================================================

# ---- GPU(s) — space-separated list passed to --gpu_use -------
GPUS="1"

# ---- Runner --------------------------------------------------
SCRIPT="run_alphapre_convlstm.py"

# ---- Experiment ----------------------------------------------
EXP_DIR="sevir"
EXP_NOTE="dawncast_pixel"
RUN_NAME="DAWNCast_pixel_sevir"

# ---- Dataset -------------------------------------------------
DATASET="sevir"                   # sevir | meteo | shanghai | cikm
IMG_SIZE=128
IMG_CHANNEL=1
FRAMES_IN=5
FRAMES_OUT=20
SEQ_LEN=25

# ---- DAWNCast: wavelet ---------------------------------------
WAVE="db6"                        # haar | db4 | db6
WAVELET_LEVEL=2                   # DWT decomposition levels J (1-4)
HF_MODE="separate"                # shared | separate

# ---- DAWNCast: Gabor LL subband (low-frequency) -------------
WEIGHT_SCALE_LOW=0.1
ALPHA_LOW=1.0
BETA_LOW=0.17
FREQ_MULTIPLIER_LOW=0.1

# ---- DAWNCast: Gabor HF subbands (high-frequency) -----------
WEIGHT_SCALE_HIGH=1.0
ALPHA_HIGH=1.0
BETA_HIGH=0.17
FREQ_MULTIPLIER_HIGH=4.0

# ---- DAWNCast: SRST Block (Spectral Temporal Refinement) ----
SPECTRAL_BLOCKS=4                 # N_g: number of groups in STR module
SPECTRAL_HIDDEN_SIZE_FACTOR=4     # rho_h: hidden expansion in STR module
SPARSITY_THRESHOLD=0.01           # soft-shrinkage lambda
CONV_KERNEL=3                     # spatial depthwise conv kernel size k

# ---- DAWNCast: general architecture -------------------------
HIDDEN_DIM=64                     # latent channel dim after lifting
SIZE_FACTOR=1.0                   # FAT Block MLP hidden expansion

# ---- Wandb ---------------------------------------------------
WANDB_STATE="online"
WANDB_PROJECT="Neurips26"



# ==============================================================
python ${SCRIPT} \
    --backbone                 dawncast \
    --seed                     0 \
    --exp_dir                  ${EXP_DIR} \
    --exp_note                 ${EXP_NOTE} \
    \
    --dataset                  ${DATASET} \
    --img_size                 ${IMG_SIZE} \
    --img_channel              ${IMG_CHANNEL} \
    --frames_in                ${FRAMES_IN} \
    --frames_out               ${FRAMES_OUT} \
    --seq_len                  ${SEQ_LEN} \
    \
    --wave                     ${WAVE} \
    --wavelet_level            ${WAVELET_LEVEL} \
    --hf_mode                  ${HF_MODE} \
    --weight_scale_low         ${WEIGHT_SCALE_LOW} \
    --alpha_low                ${ALPHA_LOW} \
    --beta_low                 ${BETA_LOW} \
    --freq_multiplier_low      ${FREQ_MULTIPLIER_LOW} \
    --weight_scale_high        ${WEIGHT_SCALE_HIGH} \
    --alpha_high               ${ALPHA_HIGH} \
    --beta_high                ${BETA_HIGH} \
    --freq_multiplier_high     ${FREQ_MULTIPLIER_HIGH} \
    --spectral_blocks          ${SPECTRAL_BLOCKS} \
    --spectral_hidden_size_factor ${SPECTRAL_HIDDEN_SIZE_FACTOR} \
    --sparsity_threshold       ${SPARSITY_THRESHOLD} \
    --conv_kernel              ${CONV_KERNEL} \
    --hidden_dim               ${HIDDEN_DIM} \
    --size_factor              ${SIZE_FACTOR} \
    \
    --wandb_state              ${WANDB_STATE} \
    --wandb_project_name       ${WANDB_PROJECT} \
    --run_name                 ${RUN_NAME} \
    --gpu_use                  ${GPUS} \
    \
    --valid \

python ${SCRIPT} \
    --backbone                 dawncast \
    --seed                     0 \
    --exp_dir                  ${EXP_DIR} \
    --exp_note                 ${EXP_NOTE} \
    \
    --dataset                  ${DATASET} \
    --img_size                 ${IMG_SIZE} \
    --img_channel              ${IMG_CHANNEL} \
    --frames_in                ${FRAMES_IN} \
    --frames_out               ${FRAMES_OUT} \
    --seq_len                  ${SEQ_LEN} \
    \
    --wave                     ${WAVE} \
    --wavelet_level            ${WAVELET_LEVEL} \
    --hf_mode                  ${HF_MODE} \
    --weight_scale_low         ${WEIGHT_SCALE_LOW} \
    --alpha_low                ${ALPHA_LOW} \
    --beta_low                 ${BETA_LOW} \
    --freq_multiplier_low      ${FREQ_MULTIPLIER_LOW} \
    --weight_scale_high        ${WEIGHT_SCALE_HIGH} \
    --alpha_high               ${ALPHA_HIGH} \
    --beta_high                ${BETA_HIGH} \
    --freq_multiplier_high     ${FREQ_MULTIPLIER_HIGH} \
    --spectral_blocks          ${SPECTRAL_BLOCKS} \
    --spectral_hidden_size_factor ${SPECTRAL_HIDDEN_SIZE_FACTOR} \
    --sparsity_threshold       ${SPARSITY_THRESHOLD} \
    --conv_kernel              ${CONV_KERNEL} \
    --hidden_dim               ${HIDDEN_DIM} \
    --size_factor              ${SIZE_FACTOR} \
    \
    --wandb_state              offline \
    --wandb_project_name       ${WANDB_PROJECT} \
    --run_name                 ${RUN_NAME} \
    --gpu_use                  ${GPUS} \
    \
    --eval \