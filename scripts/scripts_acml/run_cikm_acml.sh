#!/bin/bash
# =============================================================================
# Coordinate sweep (one axis at a time), parallel across 2 GPUs.
#
#   Phase 1 : vary hidden_dim, hold kernels & lift/proj at baseline
#   Phase 2 : vary kernel schedule, hold HD & lift/proj at baseline
#   Phase 3 : vary lift/proj options, hold HD & kernels at baseline
#
# Total jobs ≈ |HD| + |kernels| + |lift_proj@baseline_HD|  (with dedup).
# =============================================================================

# Don't use `set -e` — single config failure should not kill the sweep.

# -----------------------------------------------------------------------------
# Paths & shared config
# -----------------------------------------------------------------------------
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
SF=1.0
CONST_RATIO=0.1

GPUS=(0 1)
NUM_GPUS=${#GPUS[@]}

LOG_DIR="./sweep_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# -----------------------------------------------------------------------------
# Sweep axes
# -----------------------------------------------------------------------------
HIDDEN_DIMS=(32 64 128 256)

KERNEL_SCHEDULES=(
    "7 5 3"
    "3 5 7"
    "5 3 1"
    "1 3 5"
)

# -----------------------------------------------------------------------------
# Baselines (held fixed when other axes are swept)
# -----------------------------------------------------------------------------
BASELINE_HD=64
BASELINE_KERNELS="3 3 3"
# Baseline lift/proj per HD = the first config from get_lift_proj_configs.

# -----------------------------------------------------------------------------
# Lifting / projection schedules per hidden_dim
#   Format: "lift_dims | proj_dims"
# -----------------------------------------------------------------------------
get_lift_proj_configs() {
    local HD=$1
    case "${HD}" in
        32)
            echo "16 32 32|32 32 16 4"
            echo "8 16 32|32 16 8 4"
            ;;
        64)
            echo "16 32 64|64 32 16 4"
            echo "32 64 64|64 64 32 4"
            ;;
        128)
            echo "32 64 128|128 64 32 4"
            echo "64 128 128|128 128 64 4"
            ;;
        256)
            echo "64 128 256|256 128 64 4"
            echo "128 256 256|256 256 128 4"
            ;;
        *)
            echo "Unsupported hidden_dim=${HD}" >&2
            return 1
            ;;
    esac
}

# -----------------------------------------------------------------------------
# Run one (train + eval) experiment on a specific GPU
# -----------------------------------------------------------------------------
run_experiment() {
    local GPU=$1
    local HD=$2
    local K1=$3 K2=$4 K3=$5
    local LIFT_DIMS=$6 PROJ_DIMS=$7

    local K_TAG="k${K1}${K2}${K3}"
    local L_TAG="l${LIFT_DIMS// /_}"
    local P_TAG="p${PROJ_DIMS// /_}"
    local TAG="hd${HD}_${K_TAG}_${L_TAG}_${P_TAG}"

    echo "[GPU${GPU}] >>> START ${TAG}"

    # ---- TRAIN ----
    python3 "${RUN_FILE}" \
        --backbone              "amplinet_latent_falfcl_only_2.3.13.2.acml" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --gpu_use               ${GPU} \
        --frames_in             5 \
        --frames_out            10 \
        --exp_dir               "${EXP_DIR}" \
        --exp_note              "cikm_${TAG}" \
        --epochs                ${EPOCHS} \
        --batch_size            ${BATCH_SIZE} \
        --lr                    ${LR} \
        --wandb_state           online \
        --wandb_project_name    "${WANDB_PROJECT}" \
        --run_name              "cikm_${TAG}" \
        --ae_ckpt_path          "${AE_CKPT}" \
        --mlp_size_factor       ${SF} \
        --hidden_dim            ${HD} \
        --conv_kernel_sizes     ${K1} ${K2} ${K3} \
        --lift_dims             ${LIFT_DIMS} \
        --proj_dims             ${PROJ_DIMS} \
        --facl_const_ratio      ${CONST_RATIO} \
        --valid \
        --seed 0
    local train_rc=$?

    if [[ ${train_rc} -ne 0 ]]; then
        echo "[GPU${GPU}] !!! TRAIN FAILED (rc=${train_rc}) for ${TAG} — skipping eval"
        return ${train_rc}
    fi

    # ---- EVAL ----
    python3 "${RUN_FILE}" \
        --backbone              "amplinet_latent_falfcl_only_2.3.13.2.acml" \
        --dataset               "${DATASET}" \
        --img_size              ${IMG_SIZE} \
        --img_channel           ${IMG_CHANNEL} \
        --gpu_use               ${GPU} \
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
        --conv_kernel_sizes     ${K1} ${K2} ${K3} \
        --lift_dims             ${LIFT_DIMS} \
        --proj_dims             ${PROJ_DIMS} \
        --facl_const_ratio      ${CONST_RATIO} \
        --eval \
        --seed 0
    local eval_rc=$?

    if [[ ${eval_rc} -ne 0 ]]; then
        echo "[GPU${GPU}] !!! EVAL FAILED (rc=${eval_rc}) for ${TAG}"
    else
        echo "[GPU${GPU}] <<< DONE  ${TAG}"
    fi
    return ${eval_rc}
}

# -----------------------------------------------------------------------------
# Build the unique job list (coordinate sweep, with dedup)
# -----------------------------------------------------------------------------
declare -A SEEN
JOBS=()

add_unique_job() {
    local HD=$1 K1=$2 K2=$3 K3=$4 LIFT=$5 PROJ=$6 PHASE=$7
    local key="${HD}|${K1} ${K2} ${K3}|${LIFT}|${PROJ}"
    if [[ -z "${SEEN[$key]:-}" ]]; then
        SEEN[$key]="${PHASE}"
        JOBS+=("${HD};${K1};${K2};${K3};${LIFT};${PROJ};${PHASE}")
    fi
}

read -r BK1 BK2 BK3 <<< "${BASELINE_KERNELS}"

# Phase 1: vary HD ------------------------------------------------------------
for HD in "${HIDDEN_DIMS[@]}"; do
    LP=$(get_lift_proj_configs "${HD}" | head -n 1)
    LIFT="${LP%%|*}"; PROJ="${LP#*|}"
    add_unique_job "${HD}" "${BK1}" "${BK2}" "${BK3}" "${LIFT}" "${PROJ}" "P1_HD"
done

# Phase 2: vary kernels (at baseline HD) --------------------------------------
BASE_LP=$(get_lift_proj_configs "${BASELINE_HD}" | head -n 1)
BASE_LIFT="${BASE_LP%%|*}"; BASE_PROJ="${BASE_LP#*|}"
for KERNELS in "${KERNEL_SCHEDULES[@]}"; do
    read -r K1 K2 K3 <<< "${KERNELS}"
    add_unique_job "${BASELINE_HD}" "${K1}" "${K2}" "${K3}" "${BASE_LIFT}" "${BASE_PROJ}" "P2_K"
done

# Phase 3: vary lift/proj (at baseline HD, baseline kernels) ------------------
mapfile -t LP_CONFIGS < <(get_lift_proj_configs "${BASELINE_HD}")
for CONFIG in "${LP_CONFIGS[@]}"; do
    LIFT="${CONFIG%%|*}"; PROJ="${CONFIG#*|}"
    add_unique_job "${BASELINE_HD}" "${BK1}" "${BK2}" "${BK3}" "${LIFT}" "${PROJ}" "P3_LP"
done

# -----------------------------------------------------------------------------
# Print plan
# -----------------------------------------------------------------------------
echo "============================================================"
echo " Coordinate sweep plan"
echo "   Baseline HD      : ${BASELINE_HD}"
echo "   Baseline kernels : ${BASELINE_KERNELS}"
echo "   Baseline lift    : ${BASE_LIFT}"
echo "   Baseline proj    : ${BASE_PROJ}"
echo "   Total unique jobs: ${#JOBS[@]}"
echo "============================================================"
printf " %-7s %-4s %-9s %-15s %-15s\n" "PHASE" "HD" "KERNELS" "LIFT" "PROJ"
for job in "${JOBS[@]}"; do
    IFS=';' read -r HD K1 K2 K3 LIFT PROJ PHASE <<< "${job}"
    printf " %-7s %-4s %-9s %-15s %-15s\n" "${PHASE}" "${HD}" "${K1} ${K2} ${K3}" "${LIFT}" "${PROJ}"
done
echo "============================================================"
echo ""

# -----------------------------------------------------------------------------
# Round-robin distribute jobs into per-GPU lanes
# -----------------------------------------------------------------------------
LANE_0=()
LANE_1=()
for i in "${!JOBS[@]}"; do
    if (( i % NUM_GPUS == 0 )); then
        LANE_0+=("${JOBS[$i]}")
    else
        LANE_1+=("${JOBS[$i]}")
    fi
done

echo "GPU 0 lane: ${#LANE_0[@]} jobs"
echo "GPU 1 lane: ${#LANE_1[@]} jobs"
echo "Logs       : ${LOG_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Lane runner: process a list of jobs sequentially on one GPU
# -----------------------------------------------------------------------------
run_lane() {
    local gpu=$1
    shift
    local log_file="${LOG_DIR}/sweep_gpu${gpu}.log"
    local total=$#
    local idx=0

    {
        echo "============================================================"
        echo " Lane started on GPU ${gpu} | ${total} jobs"
        echo " Started at: $(date)"
        echo "============================================================"
    } | tee -a "${log_file}"

    for job in "$@"; do
        idx=$((idx + 1))
        IFS=';' read -r HD K1 K2 K3 LIFT_DIMS PROJ_DIMS PHASE <<< "${job}"

        {
            echo ""
            echo "------------------------------------------------------------"
            echo " [GPU${gpu}] Job ${idx}/${total} [${PHASE}] | $(date)"
            echo "   hidden_dim = ${HD}"
            echo "   kernels    = ${K1} ${K2} ${K3}"
            echo "   lift_dims  = ${LIFT_DIMS}"
            echo "   proj_dims  = ${PROJ_DIMS}"
            echo "------------------------------------------------------------"
        } | tee -a "${log_file}"

        run_experiment "${gpu}" "${HD}" "${K1}" "${K2}" "${K3}" "${LIFT_DIMS}" "${PROJ_DIMS}" \
            >> "${log_file}" 2>&1
        local rc=$?

        if [[ ${rc} -ne 0 ]]; then
            echo "[GPU${gpu}] Job ${idx}/${total} FAILED (rc=${rc}) — continuing." | tee -a "${log_file}"
        else
            echo "[GPU${gpu}] Job ${idx}/${total} ok." | tee -a "${log_file}"
        fi
    done

    {
        echo ""
        echo "============================================================"
        echo " Lane on GPU ${gpu} finished at: $(date)"
        echo "============================================================"
    } | tee -a "${log_file}"
}

# -----------------------------------------------------------------------------
# Trap so Ctrl+C kills both lanes cleanly
# -----------------------------------------------------------------------------
PIDS=()
cleanup() {
    echo ""
    echo "Caught signal — terminating sweep..."
    for pid in "${PIDS[@]}"; do
        kill "${pid}" 2>/dev/null
    done
    wait 2>/dev/null
    exit 130
}
trap cleanup SIGINT SIGTERM

# -----------------------------------------------------------------------------
# Launch lanes in parallel
# -----------------------------------------------------------------------------
echo "Launching lanes..."
echo ""

run_lane 0 "${LANE_0[@]}" &
PIDS+=($!)

run_lane 1 "${LANE_1[@]}" &
PIDS+=($!)

wait "${PIDS[@]}"

echo ""
echo "============================================================"
echo " Coordinate sweep complete."
echo " Logs: ${LOG_DIR}"
echo "============================================================"
