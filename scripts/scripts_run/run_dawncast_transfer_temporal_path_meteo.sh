#!/bin/bash
# ==============================================================
# DAWNCast — Semi-Foundation-Model Transfer (temporal path only)
# Runner  : finetune_temporal_path_transfer.py
# Backbone: DAWNCast_old  (module names match the SEVIR checkpoint)
# Space   : latent     (32×32 SD-VAE latents, 4 channels)
#
# Freezes the whole pretrained network except the temporal
# processing path inside every WaveletGaborBlock — everything
# above `reconstructed = self.idwt(...)`:
#   BandTemporalStream.gabor + .mlp + .fusion
# conv_spectral, the residual merge (x_st + gabor_residual) and
# everything downstream stay frozen. 158,436 / 59,543,144
# parameters trainable (0.27%).
#
# NOTE: the pretrained checkpoint fixes T_in=5, T_out=20, dim=64,
#       db6 / J=2 / separate. Target latents must have >= 25 frames:
#       meteo_lr_latent_32 (25) and shanghai_lr_latent_32 (25) work;
#       cikm_latent_32 only has 15 frames (5->10), so its shapes do
#       not match and the loader will raise on purpose.
# ==============================================================

# ---- GPU(s) — space-separated list passed to --gpu_use -------
GPUS="0"

# ---- Runner --------------------------------------------------
SCRIPT="finetune_temporal_path_transfer.py"

# ---- Pretrained SEVIR checkpoint to transfer from ------------
PRETRAINED_CKPT="/home/vatsal/Dataserver2/Neurips/Current_best_models/Sevir/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_sevir_lr_latent_32_config5_flow0.1_fhigh4.0_b0.17_db6_J2_separate/checkpoints/ckpt-best.pt"

# ---- Experiment ----------------------------------------------
EXP_DIR="transfer_temporal_path"
EXP_NOTE="dawncast_transfer_temporal_path_meteo"
RUN_NAME="DAWNCast_transfer_temporal_path_meteo_latent"
RUN_NAME_EVAL="${RUN_NAME}_eval"  # distinct wandb run for the test pass

# ---- Dataset -------------------------------------------------
# Latent dataset names:  meteo_lr_latent_32 | shanghai_lr_latent_32
DATASET="meteo_lr_latent_32"
TARGET_TAG="meteo"                # tags the lightweight adapter checkpoint
IMG_SIZE=32                       # latent spatial size (32×32)
IMG_CHANNEL=4                     # SD-VAE latent channels
FRAMES_IN=5
FRAMES_OUT=20
SEQ_LEN=25

# ---- Autoencoder checkpoint ----------------------------------
# Omitted on purpose: the runner picks the per-dataset frozen
# AutoencoderKL automatically. Pass --ae_ckpt_path to override.

# ---- DAWNCast: architecture (MUST match the pretrained run) --
WAVE="db6"
WAVELET_LEVEL=2
HF_MODE="separate"
WEIGHT_SCALE_LOW=0.1
ALPHA_LOW=1.0
BETA_LOW=0.17
FREQ_MULTIPLIER_LOW=0.1
WEIGHT_SCALE_HIGH=0.1
ALPHA_HIGH=1.0
BETA_HIGH=0.17
FREQ_MULTIPLIER_HIGH=4.0
SPECTRAL_BLOCKS=4
SPECTRAL_HIDDEN_SIZE_FACTOR=4
SPARSITY_THRESHOLD=0.01
CONV_KERNEL=3
HIDDEN_DIM=64
SIZE_FACTOR=1.0

# ---- Fine-tuning ---------------------------------------------
LR=1e-4
BATCH_SIZE=4
EPOCHS=50
FREEZE_CHECK_STEP=10              # assert frozen params unchanged after N steps
RESULTS_CSV="/home/vatsal/Dataserver2/Neurips/csv_files/Transfer_runs.csv"

# ---- Wandb ---------------------------------------------------
WANDB_STATE="online"
WANDB_PROJECT="Dawncast_foundation"

# ==============================================================
CUDA_VISIBLE_DEVICES=${GPUS} python ${SCRIPT} \
    --pretrained_ckpt          ${PRETRAINED_CKPT} \
    --target_tag               ${TARGET_TAG} \
    --freeze_check_step        ${FREEZE_CHECK_STEP} \
    --results_csv              ${RESULTS_CSV} \
    --csv_log_val \
    \
    --backbone                 DAWNCast_old \
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
    --lr                       ${LR} \
    --batch_size               ${BATCH_SIZE} \
    --epochs                   ${EPOCHS} \
    \
    --wandb_state              ${WANDB_STATE} \
    --wandb_project_name       ${WANDB_PROJECT} \
    --run_name                 ${RUN_NAME} \
    --gpu_use                  ${GPUS} \
    \
    --valid \

# ---- Test the best fine-tuned checkpoint + CSV row ------------
CUDA_VISIBLE_DEVICES=${GPUS} python ${SCRIPT} \
    --pretrained_ckpt          ${PRETRAINED_CKPT} \
    --target_tag               ${TARGET_TAG} \
    --results_csv              ${RESULTS_CSV} \
    \
    --backbone                 DAWNCast_old \
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
    --run_name                 ${RUN_NAME_EVAL} \
    --gpu_use                  ${GPUS} \
    \
    --eval \
