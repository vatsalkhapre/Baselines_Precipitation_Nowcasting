#!/bin/bash
# =============================================================================
# Ablation: BC vs non-BC backbone, SF=2.0 vs SF=1.0 on CIKM
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Paths & shared config
# # -----------------------------------------------------------------------------
# RUN_FILE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent_for_model_parts.py"
# AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"

# DATASET="cikm_latent_32"
# IMG_SIZE=32
# IMG_CHANNEL=4
# EXP_DIR="lastocast_minus_faclloss"
# EPOCHS=50
# BATCH_SIZE=4
# LR=1e-4
# WANDB_PROJECT="Alphapre"
# GPU=1
# HD=64
# SF=1.0

# # =============================================================================
# # Helper: train then eval for a given BACKBONE + SF
# # =============================================================================
# run_experiment() {
#     local BACKBONE=$1
#     local KERNEL=$2
#     local TAG="bb${BACKBONE}_kernel_size${KERNEL}"

#     echo "============================================================"
#     echo " TRAIN | backbone=${BACKBONE}  conv_kernel_size=${KERNEL} "
#     echo "============================================================"
#     python3 "${RUN_FILE}" \
#         --backbone              "${BACKBONE}" \
#         --dataset               "${DATASET}" \
#         --img_size              ${IMG_SIZE} \
#         --img_channel           ${IMG_CHANNEL} \
#         --gpu_use               ${GPU} \
#         --frames_in             5 \
#         --frames_out            10 \
#         --exp_dir               "${EXP_DIR}" \
#         --exp_note              "cikm_${TAG}" \
#         --epochs                ${EPOCHS} \
#         --batch_size            ${BATCH_SIZE} \
#         --lr                    ${LR} \
#         --wandb_state           online \
#         --wandb_project_name    "${WANDB_PROJECT}" \
#         --run_name              "cikm_${TAG}" \
#         --ae_ckpt_path          "${AE_CKPT}" \
#         --mlp_size_factor       ${SF} \
#         --hidden_dim            ${HD} \
#         --conv_kernel_size      ${KERNEL} \
#         --valid \
#         --seed 0

#     echo "============================================================"
#     echo " EVAL  | backbone=${BACKBONE}  mlp_size_factor=${SF}  hidden_dim=${HD}"
#     echo "============================================================"
#     python3 "${RUN_FILE}" \
#         --backbone              "${BACKBONE}" \
#         --dataset               "${DATASET}" \
#         --img_size              ${IMG_SIZE} \
#         --img_channel           ${IMG_CHANNEL} \
#         --gpu_use               ${GPU} \
#         --frames_in             5 \
#         --frames_out            10 \
#         --exp_dir               "${EXP_DIR}" \
#         --exp_note              "cikm_${TAG}" \
#         --epochs                ${EPOCHS} \
#         --batch_size            ${BATCH_SIZE} \
#         --lr                    ${LR} \
#         --wandb_state           offline \
#         --wandb_project_name    "${WANDB_PROJECT}" \
#         --run_name              "cikm_${TAG}_eval" \
#         --ae_ckpt_path          "${AE_CKPT}" \
#         --mlp_size_factor       ${SF} \
#         --hidden_dim            ${HD} \
#         --conv_kernel_size      ${KERNEL} \
#         --eval \
#         --seed 0

#     echo " Done: backbone=${BACKBONE}, mlp_size_factor=${SF}, hidden_dim=${HD}"
#     echo ""
# }


# # =============================================================================
# # Experiments
# # =============================================================================


# # 3. non-BC variant, SF=1.0
# run_experiment "amplinet_latent_falfcl_only_2.3.13.2.acml" 1
# # run_experiment "amplinet_latent_falfcl_only_2.3.13.2.acml" 3
# run_experiment "amplinet_latent_falfcl_only_2.3.13.2.acml" 5
# run_experiment "amplinet_latent_falfcl_only_2.3.13.2.acml" 7
# # run_experiment "amplinet_latent_falfcl_only_2.3.13.2.acml" 9


# echo "============================================================"
# echo " All experiments complete."
# echo "============================================================"


RUN_FILE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent_for_model_parts.py"
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"

DATASET="cikm_latent_32"
IMG_SIZE=32
IMG_CHANNEL=4
EXP_DIR="lastocast_minus_faclloss"
EPOCHS=50
BATCH_SIZE=4
LR=1e-4
WANDB_PROJECT="Alphapre"
HD=64
SF=1.0
LIFT_DIMS="32 64 64"
PROJ_DIMS="64 64 32 4"
KERNEL_SIZES="3 3 3"

# =============================================================================
# Helper: train then eval
# =============================================================================
run_experiment() {

    local GPU_ID=$1
    local BACKBONE=$2
    local CONST_RATIO=$3
    local TAG=$4

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
        --frames_out            10 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "cikm_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --gpu_use               ${GPU_ID} \
        --lr                    ${LR} \
        --wandb_state           online \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "cikm_${TAG}" \
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
        --frames_out            10 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "cikm_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --lr                    ${LR} \
        --wandb_state           offline \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "cikm_${TAG}_eval" \
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
    0.05 \
    "facl_0.05" &
PID1=$!

run_experiment \
    1 \
    "${BACKBONE}" \
    0.00 \
    "facl_0.00" &
PID1=$!
# Wait for both to finish
wait $PID1
wait $PID2



echo "============================================================"
echo " All experiments complete."
echo "============================================================"