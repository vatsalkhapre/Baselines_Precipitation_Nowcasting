#!/bin/bash
# ==============================================================
# LPCast — Pixel Space Training
# Runner  : run_alphapre_convlstm.py
# Backbone: lpcast
# Space   : pixel  (raw frames, no autoencoder)
# ==============================================================

# ---- GPU(s) — space-separated list passed to --gpu_use -------
GPUS="0"                          # e.g. "0 1 2 3" for multi-GPU

# ---- Runner --------------------------------------------------
SCRIPT="run_alphapre_convlstm.py"

# ---- Experiment ----------------------------------------------
EXP_DIR="sevir"                   # top-level experiment folder
EXP_NOTE="lpcast_pixel"           # sub-folder / run label
RUN_NAME="LPCast_pixel_sevir"     # wandb display name

# ---- Dataset -------------------------------------------------
DATASET="sevir"                   # sevir | meteo | shanghai | cikm
IMG_SIZE=128                      # spatial resolution (H = W)
IMG_CHANNEL=1                     # 1 for single-channel precip
FRAMES_IN=5
FRAMES_OUT=20
SEQ_LEN=25
STRIDE=13
BATCH_SIZE=4
NUM_WORKERS=8
PREPROCESSING=0                   # 0 = min-max normalisation

# ---- Optimiser -----------------------------------------------
EPOCHS=50

# ---- LPCast architecture ------------------------------------
# Constraints (enforced inside the model):
#   lift_dims[-1]  == HIDDEN_DIM
#   proj_dims[0]   == HIDDEN_DIM
#   proj_dims[-1]  == IMG_CHANNEL
HIDDEN_DIM=64
FACL_CONST_RATIO=0.1              # FACL loss warm-up constant ratio
MLP_SIZE_FACTOR=1.0               # AmpCell MLP hidden expansion
CONV_KERNEL_SIZES="3 3 3"         # three integers for AmpCell conv blocks
LIFT_DIMS="32 64 64"              # ends at HIDDEN_DIM (64)
PROJ_DIMS="64 64 32 4"               # starts at HIDDEN_DIM (64), ends at IMG_CHANNEL (1)

# ---- Wandb ---------------------------------------------------
WANDB_STATE="online"             # offline | online | disabled
WANDB_PROJECT="ACML"


# ==============================================================
python ${SCRIPT} \
    --backbone           lpcast \
    --seed               0 \
    --exp_dir            ${EXP_DIR} \
    --exp_note           ${EXP_NOTE} \
    \
    --dataset            ${DATASET} \
    --img_size           ${IMG_SIZE} \
    --img_channel        ${IMG_CHANNEL} \
    --frames_in          ${FRAMES_IN} \
    --frames_out         ${FRAMES_OUT} \
    --seq_len            ${SEQ_LEN} \
    --batch_size         ${BATCH_SIZE} \
    --num_workers        ${NUM_WORKERS} \
    \
    --epochs             ${EPOCHS} \
    \
    --hidden_dim         ${HIDDEN_DIM} \
    --facl_const_ratio   ${FACL_CONST_RATIO} \
    --mlp_size_factor    ${MLP_SIZE_FACTOR} \
    --conv_kernel_sizes  ${CONV_KERNEL_SIZES} \
    --lift_dims          ${LIFT_DIMS} \
    --proj_dims          ${PROJ_DIMS} \
    \
    --wandb_state        ${WANDB_STATE} \
    --wandb_project_name ${WANDB_PROJECT} \
    --run_name           ${RUN_NAME} \
    --gpu_use            ${GPUS} \
    \
    --valid \





#!/bin/bash
# ==============================================================
# LPCast — Latent Space Training
# Runner  : run_alphapre_convlstm_sevir_lr_latent.py
# Backbone: LPCast   (must match MODEL_REGISTRY key exactly)
# Space   : latent   (32×32 SD-VAE latents, 4 channels)
# ==============================================================

# ---- GPU(s) — space-separated list passed to --gpu_use -------
GPUS="0"                          # e.g. "0 1 2 3" for multi-GPU

# ---- Runner --------------------------------------------------
SCRIPT="run_alphapre_convlstm_sevir_lr_latent.py"

# ---- Experiment ----------------------------------------------
EXP_DIR="sevir_lr_latent_32"
EXP_NOTE="lpcast_latent"
RUN_NAME="LPCast_latent_sevir"

# ---- Dataset -------------------------------------------------
# Latent dataset names:  sevir_lr_latent_32 | sevir_lr_latent_32
#                        shanghai_lr_latent_32 | cikm_latent_32
DATASET="sevir_lr_latent_32"
IMG_SIZE=32                       # latent spatial size (32×32)
IMG_CHANNEL=4                     # SD-VAE latent channels
FRAMES_IN=5
FRAMES_OUT=20
SEQ_LEN=25
BATCH_SIZE=4
NUM_WORKERS=8
PREPROCESSING=0

# ---- Autoencoder checkpoint ----------------------------------
AE_CKPT_PATH="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"   # <-- SET THIS

# ---- Optimiser -----------------------------------------------
EPOCHS=50


# ---- LPCast architecture ------------------------------------
# Constraints (enforced inside the model):
#   lift_dims[-1]  == HIDDEN_DIM
#   proj_dims[0]   == HIDDEN_DIM
#   proj_dims[-1]  == IMG_CHANNEL
HIDDEN_DIM=64
FACL_CONST_RATIO=0.1
MLP_SIZE_FACTOR=1.0
CONV_KERNEL_SIZES="3 3 3"
LIFT_DIMS="32 64 64"              # ends at HIDDEN_DIM (64)
PROJ_DIMS="64 64 32 4"               # starts at HIDDEN_DIM (64), ends at IMG_CHANNEL (4)

# ---- Wandb ---------------------------------------------------
WANDB_STATE="online"
WANDB_PROJECT="ACML"



# ==============================================================
python ${SCRIPT} \
    --backbone           LPCast \
    --seed               0 \
    --exp_dir            ${EXP_DIR} \
    --exp_note           ${EXP_NOTE} \
    \
    --dataset            ${DATASET} \
    --img_size           ${IMG_SIZE} \
    --img_channel        ${IMG_CHANNEL} \
    --frames_in          ${FRAMES_IN} \
    --frames_out         ${FRAMES_OUT} \
    --seq_len            ${SEQ_LEN} \
    --batch_size         ${BATCH_SIZE} \
    --num_workers        ${NUM_WORKERS} \
    \
    --epochs             ${EPOCHS} \
    \
    --hidden_dim         ${HIDDEN_DIM} \
    --facl_const_ratio   ${FACL_CONST_RATIO} \
    --mlp_size_factor    ${MLP_SIZE_FACTOR} \
    --conv_kernel_sizes  ${CONV_KERNEL_SIZES} \
    --lift_dims          ${LIFT_DIMS} \
    --proj_dims          ${PROJ_DIMS} \
    \
    --wandb_state        ${WANDB_STATE} \
    --wandb_project_name ${WANDB_PROJECT} \
    --run_name           ${RUN_NAME} \
    --gpu_use            ${GPUS} \
    --ae_ckpt_path       ${AE_CKPT_PATH} \
    \
    --valid \

python ${SCRIPT} \
    --backbone           LPCast \
    --seed               0 \
    --exp_dir            ${EXP_DIR} \
    --exp_note           ${EXP_NOTE} \
    \
    --dataset            ${DATASET} \
    --img_size           ${IMG_SIZE} \
    --img_channel        ${IMG_CHANNEL} \
    --frames_in          ${FRAMES_IN} \
    --frames_out         ${FRAMES_OUT} \
    --seq_len            ${SEQ_LEN} \
    --batch_size         ${BATCH_SIZE} \
    --num_workers        ${NUM_WORKERS} \
    \
    --epochs             ${EPOCHS} \
    \
    --hidden_dim         ${HIDDEN_DIM} \
    --facl_const_ratio   ${FACL_CONST_RATIO} \
    --mlp_size_factor    ${MLP_SIZE_FACTOR} \
    --conv_kernel_sizes  ${CONV_KERNEL_SIZES} \
    --lift_dims          ${LIFT_DIMS} \
    --proj_dims          ${PROJ_DIMS} \
    \
    --wandb_state        "offline" \
    --wandb_project_name ${WANDB_PROJECT} \
    --run_name           ${RUN_NAME} \
    --gpu_use            ${GPUS} \
    --ae_ckpt_path       ${AE_CKPT_PATH} \
    \
    --eval \


    