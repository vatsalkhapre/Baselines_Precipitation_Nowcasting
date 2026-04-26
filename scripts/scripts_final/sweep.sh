#!/bin/bash
# ============================================================
# Generic Parameter Sweep Script
# Change SWEEP_PARAM and SWEEP_VALUES to tune any parameter.
# Each value runs on a separate GPU in parallel.
#
# USAGE:
#   bash run_sweep.sh
#
# TO CHANGE WHAT YOU'RE SWEEPING:
#   1. Set SWEEP_PARAM to the --arg name (e.g. "beta_low")
#   2. Set SWEEP_VALUES to the values you want to try
#   3. If sweeping multiple params together (e.g. beta_low AND beta_high
#      simultaneously), set SWEEP_PARAM_2 and SWEEP_VALUES_2 accordingly.
#      They will be paired positionally: value[0] with value2[0], etc.
#   4. Update the FIXED CONFIG section with your best known params.
# ============================================================

# ── What to sweep ─────────────────────────────────────────────
SWEEP_PARAM="beta_low"           # primary param arg name (without --)
SWEEP_VALUES=(1.0 10 100)        # one run per value, one GPU per value

# Optional: sweep a second param in lockstep with the first
# Set SWEEP_PARAM_2="" to disable
SWEEP_PARAM_2="beta_high"        # set to "" to sweep only one param
SWEEP_VALUES_2=(1.0 10 100)      # must match length of SWEEP_VALUES if set

# ── Dataset config — UPDATE THIS ──────────────────────────────
DATASET="cikm_latent_32"
SEQ_LEN=15
FRAMES_IN=5
FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="sweep_${SWEEP_PARAM}"

# ── Fixed config — UPDATE WITH YOUR BEST PARAMS ───────────────
BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0
WAVE="db4";        LEVEL=2;    HF_MODE="separate"
BLOCKS=1;          FACTOR=1;   K=7;          SPARSITY=0.01
WS_LOW=0.1;        WS_HIGH=0.25
A_LOW=1.0;         A_HIGH=1.0
B_LOW=1.0;         B_HIGH=1.0
F_LOW=1.0;         F_HIGH=1.0
EPOCHS=50

# ─────────────────────────────────────────────────────────────
# DO NOT EDIT BELOW THIS LINE
# ─────────────────────────────────────────────────────────────

run_experiment() {
    local GPU=$1
    local VAL=$2
    local VAL2=$3   # empty string if SWEEP_PARAM_2 not set

    # Build the sweep args dynamically
    local SWEEP_ARGS="--${SWEEP_PARAM} ${VAL}"
    if [ -n "${SWEEP_PARAM_2}" ] && [ -n "${VAL2}" ]; then
        SWEEP_ARGS="${SWEEP_ARGS} --${SWEEP_PARAM_2} ${VAL2}"
        local TAG="${SWEEP_PARAM}${VAL}_${SWEEP_PARAM_2}${VAL2}"
    else
        local TAG="${SWEEP_PARAM}${VAL}"
    fi

    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)
    local RUN_TAG="${TAG}_${WAVE}_J${LEVEL}_${HF_MODE}_b${BLOCKS}_f${FACTOR}_sp${SPARSITY}"

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | ${TAG}"
    echo "=============================================="

    # ── Train ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${RUN_TAG}" \
        --epochs ${EPOCHS} \
        --ae_ckpt_path "${AE_CKPT}" \
        --valid \
        --seq_len ${SEQ_LEN} \
        --seed ${SEED} \
        --frames_in ${FRAMES_IN} \
        --frames_out ${FRAMES_OUT} \
        --weight_scale_low ${WS_LOW} \
        --alpha_low ${A_LOW} \
        --beta_low ${B_LOW} \
        --freq_multiplier_low ${F_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${B_HIGH} \
        --freq_multiplier_high ${F_HIGH} \
        --wave ${WAVE} \
        --wavelet_level ${LEVEL} \
        --hf_mode ${HF_MODE} \
        --afno_blocks ${BLOCKS} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name "${BACKBONE}_${DS_SHORT}_${RUN_TAG}" \
        ${SWEEP_ARGS}   # ← overrides the fixed value above if same param

    # ── Eval ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${RUN_TAG}" \
        --ae_ckpt_path "${AE_CKPT}" \
        --eval \
        --seed ${SEED} \
        --seq_len ${SEQ_LEN} \
        --frames_in ${FRAMES_IN} \
        --frames_out ${FRAMES_OUT} \
        --weight_scale_low ${WS_LOW} \
        --alpha_low ${A_LOW} \
        --beta_low ${B_LOW} \
        --freq_multiplier_low ${F_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${B_HIGH} \
        --freq_multiplier_high ${F_HIGH} \
        --wave ${WAVE} \
        --wavelet_level ${LEVEL} \
        --hf_mode ${HF_MODE} \
        --afno_blocks ${BLOCKS} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'offline' \
        ${SWEEP_ARGS}

    echo "  Done: ${DS_SHORT} | ${TAG}"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# Distribute values across GPUs
# If more values than GPUs: excess runs queue on GPU 0, 1, 2...
# cyclically — each GPU runs its assigned values sequentially.
# ─────────────────────────────────────────────────────────────

N=${#SWEEP_VALUES[@]}
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "=============================================="
echo "  Sweep: --${SWEEP_PARAM} over [${SWEEP_VALUES[*]}]"
if [ -n "${SWEEP_PARAM_2}" ]; then
echo "  Also:  --${SWEEP_PARAM_2} over [${SWEEP_VALUES_2[*]}]"
fi
echo "  Dataset: ${DATASET}"
echo "  GPUs available: ${NUM_GPUS}"
echo "  Total runs: ${N}"
echo "=============================================="
echo ""

# Build per-GPU run lists
declare -A GPU_PIDS

gpu_runner() {
    local GPU=$1
    shift
    # Remaining args are index pairs: "val val2" space-separated
    while [ $# -gt 0 ]; do
        local VAL=$1
        local VAL2=$2
        shift 2
        run_experiment ${GPU} "${VAL}" "${VAL2}"
    done
}

# Assign runs to GPUs cyclically
declare -A GPU_ARGS
for (( i=0; i<N; i++ )); do
    GPU=$(( i % NUM_GPUS ))
    VAL="${SWEEP_VALUES[$i]}"
    VAL2=""
    if [ -n "${SWEEP_PARAM_2}" ]; then
        VAL2="${SWEEP_VALUES_2[$i]}"
    fi
    GPU_ARGS[$GPU]="${GPU_ARGS[$GPU]} ${VAL} ${VAL2}"
done

# Launch one background process per GPU
for GPU in "${!GPU_ARGS[@]}"; do
    gpu_runner ${GPU} ${GPU_ARGS[$GPU]} &
    GPU_PIDS[$GPU]=$!
    echo "  Launched GPU ${GPU} | runs: ${GPU_ARGS[$GPU]}"
done

echo ""

# Wait for all
for GPU in "${!GPU_PIDS[@]}"; do
    wait ${GPU_PIDS[$GPU]}
    echo "GPU ${GPU} complete!"
done

echo ""
echo "=============================================="
echo "  Sweep complete! Check wandb."
echo "=============================================="