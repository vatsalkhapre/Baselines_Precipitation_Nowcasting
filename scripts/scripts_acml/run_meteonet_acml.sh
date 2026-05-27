# run_lift_proj_sweep.sh

```bash
#!/bin/bash
# =============================================================================
# Sweep ONLY lifting/projection channel schedules
#
# Fixed:
#   hidden_dim  = 64
#   kernels     = [3,3,3]
#
# Sweep:
#   A. lift=[16 32 64]  proj=[64 32 16 4]
#   B. lift=[32 64 64]  proj=[64 64 32 4]
# =============================================================================

# Do NOT use set -e.
# One failed run should not kill the whole sweep.

# -----------------------------------------------------------------------------
# Paths & shared config
# -----------------------------------------------------------------------------
RUN_FILE="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent_for_model_parts.py"
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"

DATASET="meteo_lr_latent_32"
IMG_SIZE=32
IMG_CHANNEL=4
EXP_DIR="lastocast_minus_faclloss"
EPOCHS=50
BATCH_SIZE=4
LR=1e-4
WANDB_PROJECT="Alphapre"
SF=1.0
CONST_RATIO=0.1

# -----------------------------------------------------------------------------
# Fixed settings
# -----------------------------------------------------------------------------
HD=64
K1=3
K2=3
K3=3

# -----------------------------------------------------------------------------
# GPU configuration
# -----------------------------------------------------------------------------
GPUS=(0 1)
NUM_GPUS=${#GPUS[@]}

# -----------------------------------------------------------------------------
# Lifting / projection sweep configs
# Format:
#   "lift_dims | proj_dims"
# -----------------------------------------------------------------------------
LP_CONFIGS=(
    # "16 32 64|64 32 16 4"
    "32 64 64|64 64 32 4"
)

# -----------------------------------------------------------------------------
# Run one experiment (train + eval)
# -----------------------------------------------------------------------------
run_experiment() {

    local GPU=$1
    local LIFT_DIMS=$2
    local PROJ_DIMS=$3

    local K_TAG="k${K1}${K2}${K3}"
    local L_TAG="l${LIFT_DIMS// /_}"
    local P_TAG="p${PROJ_DIMS// /_}"

    local TAG="hd${HD}_${K_TAG}_${L_TAG}_${P_TAG}"

    echo "============================================================"
    echo "[GPU${GPU}] TRAIN"
    echo " hidden_dim = ${HD}"
    echo " kernels    = [${K1},${K2},${K3}]"
    echo " lift_dims  = [${LIFT_DIMS}]"
    echo " proj_dims  = [${PROJ_DIMS}]"
    echo "============================================================"

    # -------------------------------------------------------------------------
    # TRAIN
    # -------------------------------------------------------------------------
    python3 "${RUN_FILE}" \
        --backbone              "amplinet_latent_falfcl_only_2.3.13.2.acml" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --gpu_use               ${GPU} \
        --frames_in             5 \
        --frames_out            20 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "meteonet_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --lr                    ${LR} \
        --wandb_state           online \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "meteonet_${TAG}" \
        --ae_ckpt_path          "${AE_CKPT}" \
        --mlp_size_factor       ${SF} \
        --hidden_dim            ${HD} \
        --conv_kernel_sizes     ${K1} ${K2} ${K3} \
        --lift_dims             ${LIFT_DIMS} \
        --proj_dims             ${PROJ_DIMS} \
        --facl_const_ratio      ${CONST_RATIO} \
        --valid \
        --seed 0

    TRAIN_RC=$?

    if [[ ${TRAIN_RC} -ne 0 ]]; then
        echo "[GPU${GPU}] TRAIN FAILED for ${TAG}"
        return ${TRAIN_RC}
    fi

    echo "============================================================"
    echo "[GPU${GPU}] EVAL"
    echo "============================================================"

    # -------------------------------------------------------------------------
    # EVAL
    # -------------------------------------------------------------------------
    python3 "${RUN_FILE}" \
        --backbone              "amplinet_latent_falfcl_only_2.3.13.2.acml" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --gpu_use               ${GPU} \
        --frames_in             5 \
        --frames_out            20 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "meteonet_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --lr                    ${LR} \
        --wandb_state           offline \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "meteonet_${TAG}_eval" \
        --ae_ckpt_path          "${AE_CKPT}" \
        --mlp_size_factor       ${SF} \
        --hidden_dim            ${HD} \
        --conv_kernel_sizes     ${K1} ${K2} ${K3} \
        --lift_dims             ${LIFT_DIMS} \
        --proj_dims             ${PROJ_DIMS} \
        --facl_const_ratio      ${CONST_RATIO} \
        --eval \
        --seed 0

    EVAL_RC=$?

    if [[ ${EVAL_RC} -ne 0 ]]; then
        echo "[GPU${GPU}] EVAL FAILED for ${TAG}"
    else
        echo "[GPU${GPU}] DONE ${TAG}"
    fi
}

# -----------------------------------------------------------------------------
# Launch jobs
# -----------------------------------------------------------------------------
PIDS=()

for i in "${!LP_CONFIGS[@]}"; do

    GPU=${GPUS[$((i % NUM_GPUS))]}

    CONFIG="${LP_CONFIGS[$i]}"

    LIFT_DIMS="${CONFIG%%|*}"
    PROJ_DIMS="${CONFIG#*|}"

    run_experiment "${GPU}" "${LIFT_DIMS}" "${PROJ_DIMS}" &

    PIDS+=($!)

done

# -----------------------------------------------------------------------------
# Wait for all jobs
# -----------------------------------------------------------------------------
FAIL=0

for pid in "${PIDS[@]}"; do
    wait ${pid} || FAIL=1
done

# -----------------------------------------------------------------------------
# Final status
# -----------------------------------------------------------------------------
if [[ ${FAIL} -ne 0 ]]; then
    echo "============================================================"
    echo " Some experiments failed."
    echo "============================================================"
else
    echo "============================================================"
    echo " All experiments completed successfully."
    echo "============================================================"
fi
```
