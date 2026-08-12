#!/bin/bash
# ==============================================================
# DAWNCast — Transfer adaptation-surface sweep
# Runner  : finetune_temporal_path_transfer.py
# Backbone: DAWNCast_old  (module names match the SEVIR checkpoint)
#
# One run = one (dataset, unfreeze-surface, lr) triple. Each writes a
# single final test row to ${RESULTS_CSV} (no per-validation rows).
#
#   usage: bash run_dawncast_transfer_sweep.sh <dataset> <config> <gpu> [lr] [train_frac]
#     dataset    : meteo | shanghai
#     config     : zeroshot | temporal | liftproj | normbias
#                | normbias_stem | liftprojonly | dwspatial
#     gpu        : CUDA device index
#     lr         : optional, default 1e-4
#     train_frac : optional, default 1.0 (fraction of the TRAINING set)
#
# Surfaces (all except zeroshot/liftprojonly keep the temporal path open):
#   temporal      gabor + mlp + fusion                       158,496
#   liftproj      temporal + lifting + projection            425,124
#   normbias      temporal + GroupNorm affine + all biases   249,764
#   normbias_stem temporal + norms/biases of lift+proj+temporal
#                 only, conv_spectral excluded               160,280
#   liftprojonly  lifting + projection, temporal FROZEN      266,628
#   dwspatial     temporal + depthwise spatial convs         204,576
# ==============================================================
set -euo pipefail

DATASET_ARG=${1:?dataset: meteo|shanghai}
CONFIG=${2:?config: zeroshot|temporal|liftproj|normbias|normbias_stem|liftprojonly|dwspatial}
GPUS=${3:?gpu index}
LR=${4:-1e-4}
TRAIN_FRAC=${5:-1.0}

SCRIPT="finetune_temporal_path_transfer.py"
PRETRAINED_CKPT="/home/vatsal/Dataserver2/Neurips/Current_best_models/Sevir/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_sevir_lr_latent_32_config5_flow0.1_fhigh4.0_b0.17_db6_J2_separate/checkpoints/ckpt-best.pt"
RESULTS_CSV="/home/vatsal/Dataserver2/Neurips/csv_files/Transfer_runs.csv"

# ---- Dataset ------------------------------------------------
case ${DATASET_ARG} in
  meteo)    DATASET="meteo_lr_latent_32" ;;
  shanghai) DATASET="shanghai_lr_latent_32" ;;
  *) echo "unknown dataset: ${DATASET_ARG}"; exit 1 ;;
esac

# ---- Adaptation surface -------------------------------------
ZERO_SHOT=""
case ${CONFIG} in
  zeroshot)      UNFREEZE="temporal";                    ZERO_SHOT="--zero_shot" ;;
  temporal)      UNFREEZE="temporal" ;;
  liftproj)      UNFREEZE="temporal lifting projection" ;;
  normbias)      UNFREEZE="temporal norms biases" ;;
  normbias_stem) UNFREEZE="temporal norms_stem biases_stem" ;;
  liftprojonly)  UNFREEZE="lifting projection" ;;
  dwspatial)     UNFREEZE="temporal dw_spatial" ;;
  *) echo "unknown config: ${CONFIG}"; exit 1 ;;
esac

# ---- Experiment naming (unique per dataset/config/lr/frac) --
EXP_DIR="transfer_sweep"
LR_TAG=$(echo ${LR} | tr -d ' ')
FRAC_TAG=""
if [ "${TRAIN_FRAC}" != "1.0" ]; then
    FRAC_TAG="_frac$(python -c "print(int(float('${TRAIN_FRAC}')*100))")"
fi
EXP_NOTE="transfer_${CONFIG}_${DATASET_ARG}_lr${LR_TAG}${FRAC_TAG}"
RUN_NAME="DAWNCast_transfer_${CONFIG}_${DATASET_ARG}_lr${LR_TAG}${FRAC_TAG}"
TARGET_TAG="${DATASET_ARG}_${CONFIG}${FRAC_TAG}"

# ---- Architecture (MUST match the pretrained run) -----------
WAVE="db6"; WAVELET_LEVEL=2; HF_MODE="separate"
WEIGHT_SCALE_LOW=0.1;  ALPHA_LOW=1.0;  BETA_LOW=0.17;  FREQ_MULTIPLIER_LOW=0.1
WEIGHT_SCALE_HIGH=0.1; ALPHA_HIGH=1.0; BETA_HIGH=0.17; FREQ_MULTIPLIER_HIGH=4.0
SPECTRAL_BLOCKS=4; SPECTRAL_HIDDEN_SIZE_FACTOR=4; SPARSITY_THRESHOLD=0.01
CONV_KERNEL=3; HIDDEN_DIM=64; SIZE_FACTOR=1.0

# ---- Data / training ----------------------------------------
IMG_SIZE=32; IMG_CHANNEL=4; FRAMES_IN=5; FRAMES_OUT=20; SEQ_LEN=25
BATCH_SIZE=4; EPOCHS=50; FREEZE_CHECK_STEP=10

# ---- Wandb ---------------------------------------------------
WANDB_STATE="online"
WANDB_PROJECT="Dawncast_foundation"

COMMON=(
  --pretrained_ckpt ${PRETRAINED_CKPT}
  --unfreeze ${UNFREEZE}
  --target_tag ${TARGET_TAG}
  --results_csv ${RESULTS_CSV}
  --backbone DAWNCast_old
  --seed 0
  --exp_dir ${EXP_DIR}
  --exp_note ${EXP_NOTE}
  --dataset ${DATASET}
  --img_size ${IMG_SIZE} --img_channel ${IMG_CHANNEL}
  --frames_in ${FRAMES_IN} --frames_out ${FRAMES_OUT} --seq_len ${SEQ_LEN}
  --wave ${WAVE} --wavelet_level ${WAVELET_LEVEL} --hf_mode ${HF_MODE}
  --weight_scale_low ${WEIGHT_SCALE_LOW} --alpha_low ${ALPHA_LOW}
  --beta_low ${BETA_LOW} --freq_multiplier_low ${FREQ_MULTIPLIER_LOW}
  --weight_scale_high ${WEIGHT_SCALE_HIGH} --alpha_high ${ALPHA_HIGH}
  --beta_high ${BETA_HIGH} --freq_multiplier_high ${FREQ_MULTIPLIER_HIGH}
  --spectral_blocks ${SPECTRAL_BLOCKS}
  --spectral_hidden_size_factor ${SPECTRAL_HIDDEN_SIZE_FACTOR}
  --sparsity_threshold ${SPARSITY_THRESHOLD} --conv_kernel ${CONV_KERNEL}
  --hidden_dim ${HIDDEN_DIM} --size_factor ${SIZE_FACTOR}
  --lr ${LR} --batch_size ${BATCH_SIZE} --train_frac ${TRAIN_FRAC}
  --wandb_state ${WANDB_STATE} --wandb_project_name ${WANDB_PROJECT}
  --gpu_use ${GPUS}
)

if [ -n "${ZERO_SHOT}" ]; then
    # No training: evaluate the pretrained checkpoint on the target test set.
    CUDA_VISIBLE_DEVICES=${GPUS} python ${SCRIPT} "${COMMON[@]}" \
        --epochs 1 --run_name ${RUN_NAME} --zero_shot
    exit 0
fi

# ---- Train ---------------------------------------------------
CUDA_VISIBLE_DEVICES=${GPUS} python ${SCRIPT} "${COMMON[@]}" \
    --epochs ${EPOCHS} \
    --freeze_check_step ${FREEZE_CHECK_STEP} \
    --run_name ${RUN_NAME} \
    --valid

# ---- Test the best checkpoint -> final CSV row ---------------
CUDA_VISIBLE_DEVICES=${GPUS} python ${SCRIPT} "${COMMON[@]}" \
    --epochs ${EPOCHS} \
    --run_name ${RUN_NAME}_eval \
    --eval
