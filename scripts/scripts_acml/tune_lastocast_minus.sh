#!/bin/bash
# =============================================================================
# Hyperparameter Tuning: mlp_size_factor x hidden_dim on MeteoNet
# Backbone: amplinet_latent_falfcl_only_2.3.13.2.mse.BC
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
RUN_FILE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent_for_model_parts.py"
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"

# -----------------------------------------------------------------------------
# Fixed config
# -----------------------------------------------------------------------------
BACKBONE="amplinet_latent_falfcl_only_2.3.13.2.mse.BC"
DATASET="meteo_lr_latent_32"
IMG_SIZE=32
IMG_CHANNEL=4
EXP_DIR="2_3_13_2_mse_BC_meteo_hparam_tuning"
EPOCHS=50
BATCH_SIZE=4
LR=1e-4
WARMUP=1000
SCHEDULER="cosine"
WANDB_PROJECT="Alphapre"
WANDB_STATE="online"
GPU=1

# -----------------------------------------------------------------------------
# Hyperparameter grid
# -----------------------------------------------------------------------------
MLP_SIZE_FACTORS=(1.0 2.0 4.0)
HIDDEN_DIMS=(64 128 256)

# =============================================================================
# Grid search loop
# =============================================================================
for SF in "${MLP_SIZE_FACTORS[@]}"; do
    for HD in "${HIDDEN_DIMS[@]}"; do

        # Unique tag for this combination — used in exp_note, run_name, logs
        TAG="sf${SF}_hd${HD}"

        echo "============================================================"
        echo " TRAIN | mlp_size_factor=${SF}  hidden_dim=${HD}"
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
            --wandb_state           ${WANDB_STATE} \
            --wandb_project_name    "${WANDB_PROJECT}" \
            --run_name              "meteo_${TAG}" \
            --ae_ckpt_path          "${AE_CKPT}" \
            --mlp_size_factor       ${SF} \
            --hidden_dim            ${HD} \
            --valid \
            --seed 0

        echo "============================================================"
        echo " EVAL  | mlp_size_factor=${SF}  hidden_dim=${HD}"
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

        echo " Done: mlp_size_factor=${SF}, hidden_dim=${HD}"
        echo ""

    done
done

echo "============================================================"
echo " All tuning runs complete. ${#MLP_SIZE_FACTORS[@]} x ${#HIDDEN_DIMS[@]} = $((${#MLP_SIZE_FACTORS[@]} * ${#HIDDEN_DIMS[@]})) combinations."
echo "============================================================"