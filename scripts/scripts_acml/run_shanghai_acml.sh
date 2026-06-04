

RUN_FILE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent_for_model_parts.py"
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth"

DATASET="shanghai_lr_latent_32"
IMG_SIZE=32
IMG_CHANNEL=4
EXP_DIR="lastocast_minus_faclloss"
EPOCHS=50
BATCH_SIZE=4
LR=1e-4
WANDB_PROJECT="Alphapre"
SF=1.0
CONST_RATIO=0.1
KERNEL_SIZES="3 3 3"

# =============================================================================
# Helper: train then eval
# =============================================================================
run_experiment() {

    local GPU_ID=$1
    local BACKBONE=$2
    local HD=$3
    local LIFT_DIMS=$4
    local PROJ_DIMS=$5
    local TAG=$6

    echo "============================================================"
    echo " [GPU${GPU_ID}] TRAIN | backbone=${BACKBONE}"
    echo " lift_dims = ${LIFT_DIMS}"
    echo " proj_dims = ${PROJ_DIMS}"
    echo "============================================================"

    python3 "${RUN_FILE}" \
        --backbone              "${BACKBONE}" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --frames_in             5 \
        --frames_out            20 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "shanghai_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --gpu_use               ${GPU_ID} \
        --lr                    ${LR} \
        --wandb_state           offline \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "shanghai_${TAG}" \
        --ae_ckpt_path          "${AE_CKPT}" \
        --mlp_size_factor       ${SF} \
        --hidden_dim            ${HD} \
        --conv_kernel_sizes     ${KERNEL_SIZES} \
        --lift_dims             ${LIFT_DIMS} \
        --proj_dims             ${PROJ_DIMS} \
        --facl_const_ratio      ${CONST_RATIO} \
        --valid \
        --seed 0

    echo "============================================================"
    echo " [GPU${GPU_ID}] EVAL | backbone=${BACKBONE}"
    echo "============================================================"

    python3 "${RUN_FILE}" \
        --backbone              "${BACKBONE}" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --gpu_use               ${GPU_ID} \
        --frames_in             5 \
        --frames_out            20 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "shanghai_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --lr                    ${LR} \
        --wandb_state           offline \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "shanghai_${TAG}_eval" \
        --ae_ckpt_path          "${AE_CKPT}" \
        --mlp_size_factor       ${SF} \
        --hidden_dim            ${HD} \
        --conv_kernel_sizes     ${KERNEL_SIZES} \
        --lift_dims             ${LIFT_DIMS} \
        --proj_dims             ${PROJ_DIMS} \
        --facl_const_ratio      ${CONST_RATIO} \
        --eval \
        --seed 0

    echo "[GPU${GPU_ID}] Done: ${TAG}"
    echo ""
}

# =============================================================================
# Experiments
# =============================================================================

BACKBONE="amplinet_latent_falfcl_only_2.3.13.2.acml"

# --------------------------------------------------
# First pair: GPU 0 and GPU 1 in parallel
# --------------------------------------------------
echo "Starting first pair..."

run_experiment \
    0 \
    "${BACKBONE}" \
    128 \
    "64 128 128" \
    "128 128 64 4" \
    "HD_128_64128128" &
PID1=$!

# wait $PID1

run_experiment \
    0 \
    "${BACKBONE}" \
    128 \
    "128 128 128" \
    "128 128 128 4" \
    "HD_128_128128128" &
PID1=$!

wait $PID1

# --------------------------------------------------
# Second pair: GPU 0 and GPU 1 in parallel
# --------------------------------------------------
echo "Starting second pair..."


