#!/bin/bash
# ============================================================
# DAWNCast (old block) — Meteonet Gabor frequency sweep, PIXEL SPACE
#
# Runner   : run_alphapre_convlstm.py
# Backbone : DAWNCast_old  -> models/DAWNCast/dawncast_old.py
# Space    : pixel (raw 128x128 frames, no autoencoder)
#
# Sweeps the (freq_multiplier_low, freq_multiplier_high) grid.
# Jobs are round-robined over the GPUS list; each GPU runs its
# share sequentially, all GPUs run in parallel.
# ============================================================

RUNNER="run_alphapre_convlstm.py"
BACKBONE="DAWNCast"

# ---- GPUs to use (one worker per entry) ----------------------
GPUS=(0 1)

# ---- Dataset (Meteonet, pixel space) -------------------------
DATASET="meteo"
IMG_SIZE=128
IMG_CHANNEL=1
SEQ_LEN=25
FRAMES_IN=5
FRAMES_OUT=20
BATCH_SIZE=4

# ---- Experiment ---------------------------------------------
EXP_DIR="gabor_exp_meteo_pixel"
EPOCHS=50
SEED=0

# ---- Wavelet -------------------------------------------------
WAVE="db6"
WAVELET_LEVEL=1
HF_MODE="separate"

# ---- SRST / spectral block -----------------------------------
SPECTRAL_BLOCKS=4                 # latent runner called this --afno_blocks
SPECTRAL_HIDDEN_SIZE_FACTOR=4     # ... --afno2D_hidden_size_factor
SPARSITY_THRESHOLD=0.01           # ... --afno_sparsity_threshold
CONV_KERNEL=3

# ---- General architecture ------------------------------------
HIDDEN_DIM=64
SIZE_FACTOR=1.0

# ---- Gabor (fixed across the sweep) --------------------------
WS_LOW=0.1
WS_HIGH=1.0
A_LOW=1.0
A_HIGH=1.0
B_LOW=0.0995
B_HIGH=0.1643

# ---- Wandb ---------------------------------------------------
# 'disabled' -> no logging at all (no online sync, no offline run dirs).
# Set to 'online' / 'offline' to re-enable.
WANDB_STATE="online"
WANDB_PROJECT="DAWNCAST_Gabor_sweep"

# ============================================================
run_experiment() {
    local GPU=$1
    local F_LOW=$2
    local F_HIGH=$3

    local TAG="Meteonet_pixel_flow${F_LOW}_fhigh${F_HIGH}"
    local RUN_NAME="${BACKBONE}_${DATASET}_${TAG}"

    echo "=============================================="
    echo "GPU ${GPU} | Meteonet pixel | flow=${F_LOW} | fhigh=${F_HIGH}"
    echo "=============================================="

    # ---------------- train ----------------
    CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
        --backbone                      ${BACKBONE} \
        --seed                          ${SEED} \
        --exp_dir                       ${EXP_DIR} \
        --exp_note                      "${TAG}" \
        --epochs                        ${EPOCHS} \
        \
        --dataset                       ${DATASET} \
        --img_size                      ${IMG_SIZE} \
        --img_channel                   ${IMG_CHANNEL} \
        --seq_len                       ${SEQ_LEN} \
        --frames_in                     ${FRAMES_IN} \
        --frames_out                    ${FRAMES_OUT} \
        --batch_size                    ${BATCH_SIZE} \
        --num_workers                   8 \
        \
        --wave                          ${WAVE} \
        --wavelet_level                 ${WAVELET_LEVEL} \
        --hf_mode                       ${HF_MODE} \
        --weight_scale_low              ${WS_LOW} \
        --alpha_low                     ${A_LOW} \
        --beta_low                      ${B_LOW} \
        --freq_multiplier_low           ${F_LOW} \
        --weight_scale_high             ${WS_HIGH} \
        --alpha_high                    ${A_HIGH} \
        --beta_high                     ${B_HIGH} \
        --freq_multiplier_high          ${F_HIGH} \
        --spectral_blocks               ${SPECTRAL_BLOCKS} \
        --spectral_hidden_size_factor   ${SPECTRAL_HIDDEN_SIZE_FACTOR} \
        --sparsity_threshold            ${SPARSITY_THRESHOLD} \
        --conv_kernel                   ${CONV_KERNEL} \
        --hidden_dim                    ${HIDDEN_DIM} \
        --size_factor                   ${SIZE_FACTOR} \
        \
        --wandb_state                   ${WANDB_STATE} \
        --wandb_project_name            ${WANDB_PROJECT} \
        --run_name                      "${RUN_NAME}" \
        --gpu_use                       ${GPU} \
        --valid

    # ---------------- eval ----------------
    CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
        --backbone                      ${BACKBONE} \
        --seed                          ${SEED} \
        --exp_dir                       ${EXP_DIR} \
        --exp_note                      "${TAG}" \
        \
        --dataset                       ${DATASET} \
        --img_size                      ${IMG_SIZE} \
        --img_channel                   ${IMG_CHANNEL} \
        --seq_len                       ${SEQ_LEN} \
        --frames_in                     ${FRAMES_IN} \
        --frames_out                    ${FRAMES_OUT} \
        --batch_size                    ${BATCH_SIZE} \
        --num_workers                   8 \
        \
        --wave                          ${WAVE} \
        --wavelet_level                 ${WAVELET_LEVEL} \
        --hf_mode                       ${HF_MODE} \
        --weight_scale_low              ${WS_LOW} \
        --alpha_low                     ${A_LOW} \
        --beta_low                      ${B_LOW} \
        --freq_multiplier_low           ${F_LOW} \
        --weight_scale_high             ${WS_HIGH} \
        --alpha_high                    ${A_HIGH} \
        --beta_high                     ${B_HIGH} \
        --freq_multiplier_high          ${F_HIGH} \
        --spectral_blocks               ${SPECTRAL_BLOCKS} \
        --spectral_hidden_size_factor   ${SPECTRAL_HIDDEN_SIZE_FACTOR} \
        --sparsity_threshold            ${SPARSITY_THRESHOLD} \
        --conv_kernel                   ${CONV_KERNEL} \
        --hidden_dim                    ${HIDDEN_DIM} \
        --size_factor                   ${SIZE_FACTOR} \
        \
        --wandb_state                   ${WANDB_STATE} \
        --wandb_project_name            ${WANDB_PROJECT} \
        --run_name                      "${RUN_NAME}" \
        --gpu_use                       ${GPU} \
        --eval
}

# -------------------------------------------------
# Sweep grid
# -------------------------------------------------
FLOWS=(1.09 3.28 8.74 17.49 34.34)
FHIGHS=(0.14 0.42 1.12 2.25 4.41)

# Pairs to skip, e.g. SKIP_PAIRS=("1.09 0.14" "3.28 0.42")
SKIP_PAIRS=()

TASKS=()
for F_LOW in "${FLOWS[@]}"; do
    for F_HIGH in "${FHIGHS[@]}"; do
        skip=0
        for pair in "${SKIP_PAIRS[@]}"; do
            if [[ "$pair" == "$F_LOW $F_HIGH" ]]; then
                skip=1
                break
            fi
        done
        (( skip )) && continue
        TASKS+=("$F_LOW $F_HIGH")
    done
done

NUM_GPUS=${#GPUS[@]}

# One worker per GPU; worker `slot` takes tasks slot, slot+NUM_GPUS, ...
run_worker() {
    local SLOT=$1
    local GPU=${GPUS[$SLOT]}
    for ((i=SLOT; i<${#TASKS[@]}; i+=NUM_GPUS)); do
        read F_LOW F_HIGH <<< "${TASKS[$i]}"
        run_experiment "$GPU" "$F_LOW" "$F_HIGH"
    done
}

echo "=============================================="
echo "${#TASKS[@]} job(s) over ${NUM_GPUS} GPU(s): ${GPUS[*]}"
for ((s=0; s<NUM_GPUS; s++)); do
    n=0
    for ((i=s; i<${#TASKS[@]}; i+=NUM_GPUS)); do n=$((n+1)); done
    echo "  GPU ${GPUS[$s]} -> ${n} job(s)"
done
echo "=============================================="

PIDS=()
for ((s=0; s<NUM_GPUS; s++)); do
    run_worker "$s" &
    PIDS+=($!)
done

for ((s=0; s<NUM_GPUS; s++)); do
    wait "${PIDS[$s]}"
    echo "GPU ${GPUS[$s]} finished."
done

echo "All Meteonet pixel-space experiments completed."
