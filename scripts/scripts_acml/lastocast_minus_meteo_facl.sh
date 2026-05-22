#!/bin/bash
# =============================================================================
# Ablation: BC vs non-BC backbone, SF=2.0 vs SF=1.0 on MeteoNet
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Paths & shared config  — set once at the top
# -----------------------------------------------------------------------------
RUN_FILE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent_for_model_parts.py"
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"

DATASET="meteo_lr_latent_32"
IMG_SIZE=32
IMG_CHANNEL=4
EXP_DIR="lastocast_minus_faclloss"
EPOCHS=50
BATCH_SIZE=4
LR=1e-4               # ← was missing entirely; caused crash on --lr ${LR}
WANDB_PROJECT="Alphapre"
GPU=1
HD=64


# =============================================================================
# Helper: train then eval for a given BACKBONE + SF
# =============================================================================
run_experiment() {
    local BACKBONE=$1
    local SF=$2
    local TAG="sf${SF}_bb${BACKBONE}"

    echo "============================================================"
    echo " TRAIN | backbone=${BACKBONE}  mlp_size_factor=${SF}  hidden_dim=${HD}"
    echo "============================================================"
    python3 "${RUN_FILE}" \
        --backbone              "${BACKBONE}" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --gpu_use               ${GPU} \
        --frames_in             5 \
        --frames_out            20 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "meteo_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --lr                    ${LR} \
        --wandb_state           online \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "meteo_${TAG}" \
        --ae_ckpt_path          "${AE_CKPT}" \
        --mlp_size_factor       ${SF} \
        --hidden_dim            ${HD} \
        --valid \
        --seed 0

    echo "============================================================"
    echo " EVAL  | backbone=${BACKBONE}  mlp_size_factor=${SF}  hidden_dim=${HD}"
    echo "============================================================"
    python3 "${RUN_FILE}" \
        --backbone              "${BACKBONE}" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --gpu_use               ${GPU} \
        --frames_in             5 \
        --frames_out            20 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "meteo_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --lr                    ${LR} \
        --wandb_state           offline \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "meteo_${TAG}_eval" \
        --ae_ckpt_path          "${AE_CKPT}" \
        --mlp_size_factor       ${SF} \
        --hidden_dim            ${HD} \
        --eval \
        --seed 0

    echo " Done: backbone=${BACKBONE}, mlp_size_factor=${SF}, hidden_dim=${HD}"
    echo ""
}


# =============================================================================
# Experiments
# =============================================================================

# 1. BC variant,     SF=2.0
run_experiment "amplinet_latent_falfcl_only_2.3.13.2.BC" 2.0

# 1.5. BC variant,     SF=4.0
run_experiment "amplinet_latent_falfcl_only_2.3.13.2.BC" 4.0


# 1.75. non-BC variant, SF=4.0
run_experiment "amplinet_latent_falfcl_only_2.3.13.2" 4.0


# 2. non-BC variant, SF=2.0
run_experiment "amplinet_latent_falfcl_only_2.3.13.2" 2.0

# 3. non-BC variant, SF=1.0
run_experiment "amplinet_latent_falfcl_only_2.3.13.2" 1.0


echo "============================================================"
echo " All experiments complete."
echo "============================================================"