#!/bin/bash
# ==============================================================
# DAWNCast — Latent Space Training
# Runner  : run_alphapre_convlstm_sevir_lr_latent.py
# Backbone: DAWNCast   (must match MODEL_REGISTRY key exactly)
# Space   : latent     (32×32 SD-VAE latents, 4 channels)
# ==============================================================

# ---- GPU(s) — space-separated list passed to --gpu_use -------
GPUS="0"

# ---- Runner --------------------------------------------------
SCRIPT="run_alphapre_convlstm_sevir_lr_latent.py"

# ---- Experiment ----------------------------------------------
EXP_DIR="sevir_lr_latent_32"
EXP_NOTE="dawncast_latent"
RUN_NAME="DAWNCast_latent_sevir"

# ---- Dataset -------------------------------------------------
# Latent dataset names:  sevir_lr_latent_32 | meteo_lr_latent_32
#                        shanghai_lr_latent_32 | cikm_latent_32
DATASET="sevir_lr_latent_32"
IMG_SIZE=32                       # latent spatial size (32×32)
IMG_CHANNEL=4                     # SD-VAE latent channels
FRAMES_IN=5
FRAMES_OUT=20
SEQ_LEN=25


# ---- Autoencoder checkpoint ----------------------------------
AE_CKPT_PATH="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"  # <-- SET THIS



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

# ---- DAWNCast: SRST Block -----------------------------------
SPECTRAL_BLOCKS=4                 # N_g: number of groups in STR module
SPECTRAL_HIDDEN_SIZE_FACTOR=4     # rho_h: hidden expansion in STR module
SPARSITY_THRESHOLD=0.01           # soft-shrinkage lambda
CONV_KERNEL=3                     # spatial depthwise conv kernel size k

# ---- DAWNCast: general architecture -------------------------
HIDDEN_DIM=64
SIZE_FACTOR=1.0

# ---- Wandb ---------------------------------------------------
WANDB_STATE="offline"
WANDB_PROJECT="Neurips26"


# ==============================================================
python ${SCRIPT} \
    --backbone                 DAWNCast \
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
    --ae_ckpt_path             ${AE_CKPT_PATH} \
    \
    --valid \