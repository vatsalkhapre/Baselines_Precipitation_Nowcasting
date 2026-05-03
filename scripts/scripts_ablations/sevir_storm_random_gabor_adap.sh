#!/bin/bash
# ============================================================
# SEVIR — Config 2 + Config 3 (parallel)
# GPU 0 → Storm_configA & Storm_configC
# GPU 1 → Random_configA & Random_configC
# ============================================================

RUNNER="run_sevir_storm_random_latent_model_novel_ablations.py"
BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Dataset ───────────────────────────────────────────────────
DATASET="sevir_lr_latent_32"
SEQ_LEN=25; FRAMES_IN=5; FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"
EPOCHS=50
DS_SHORT="sevir"

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local CFG_NAME=$2
    local EXP_DIR=$3
    local WAVE=$4; local LEVEL=$5; local HF_MODE=$6
    local BLOCKS=$7; local FACTOR=$8; local K=$9
    local SPARSITY=${10}
    local WS_LOW=${11}; local WS_HIGH=${12}
    local A_LOW=${13};  local A_HIGH=${14}
    local B_LOW=${15};  local B_HIGH=${16}
    local F_LOW=${17};  local F_HIGH=${18}
    local SEVIR_TYPE=${19}

    local TAG="${CFG_NAME}_flow${F_LOW}_fhigh${F_HIGH}_b${B_LOW}_${WAVE}_J${LEVEL}_${HF_MODE}"

    echo "=============================================="
    echo "  GPU ${GPU} | SEVIR | ${CFG_NAME}"
    echo "  wave=${WAVE} J=${LEVEL} blocks=${BLOCKS} factor=${FACTOR}"
    echo "  f_low=${F_LOW} f_high=${F_HIGH} beta=${B_LOW}"
    echo "=============================================="

    # ── Train ──
    # CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
    #     --backbone ${BACKBONE} \
    #     --dataset ${DATASET} \
    #     --exp_dir ${EXP_DIR} \
    #     --exp_note "${TAG}" \
    #     --epochs ${EPOCHS} \
    #     --ae_ckpt_path "${AE_CKPT}" \
    #     --valid \
    #     --seq_len ${SEQ_LEN} \
    #     --seed ${SEED} \
    #     --frames_in ${FRAMES_IN} \
    #     --frames_out ${FRAMES_OUT} \
    #     --weight_scale_low ${WS_LOW} \
    #     --alpha_low ${A_LOW} \
    #     --beta_low ${B_LOW} \
    #     --freq_multiplier_low ${F_LOW} \
    #     --weight_scale_high ${WS_HIGH} \
    #     --alpha_high ${A_HIGH} \
    #     --beta_high ${B_HIGH} \
    #     --freq_multiplier_high ${F_HIGH} \
    #     --wave ${WAVE} \
    #     --wavelet_level ${LEVEL} \
    #     --hf_mode ${HF_MODE} \
    #     --afno_blocks ${BLOCKS} \
    #     --sevir_dataset_type ${SEVIR_TYPE} \
    #     --afno2D_hidden_size_factor ${FACTOR} \
    #     --afno_sparsity_threshold ${SPARSITY} \
    #     --conv_kernel ${K} \
    #     --num_workers 8 \
    #     --wandb_state 'offline' \
    #     --wandb_project_name 'Nowcasting_ablations' \
    #     --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

    # ── Eval ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${TAG}" \
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
        --sevir_dataset_type ${SEVIR_TYPE} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  Done: SEVIR | ${CFG_NAME}"
    echo ""
}

echo "=============================================="
echo "  Starting SEVIR Parallel Runs"
echo "  GPU 0 → Storm_configA & Storm_configC"
echo "  GPU 1 → Random_configA & Random_configC"
echo "=============================================="
echo ""

# ─────────────────────────────────────────────────────────────
# GPU 0 Runs
# ─────────────────────────────────────────────────────────────
# run_experiment 0 "Storm_configC" "Sevir_Storm" \
#     db6 2 separate \
#     4 4 3 0.01 \
#     0.1 1.0 \
#     1.0 1.0 \
#     0.17 0.17 \
#     4.0 4.0 \
#     'storm' &
# PID_STORM_C=$!

# run_experiment 0 "Storm_configA" "Sevir_Storm" \
#     db6 2 separate \
#     4 4 3 0.01 \
#     0.1 1.0 \
#     1.0 1.0 \
#     100 100 \
#     0.1 0.1 \
#     'storm' &
# PID_STORM_A=$!

# ─────────────────────────────────────────────────────────────
# GPU 1 Runs
# ─────────────────────────────────────────────────────────────

# run_experiment 1 "Random_configC" "Sevir_Random" \
#     db6 2 separate \
#     4 4 3 0.01 \
#     0.1 1.0 \
#     1.0 1.0 \
#     0.17 0.17 \
#     4.0 4.0 \
#     'random' &
# PID_RANDOM_C=$!

run_experiment 2 "Random_configA" "Sevir_Random" \
    db6 2 separate \
    4 4 3 0.01 \
    0.1 1.0 \
    1.0 1.0 \
    100 100 \
    0.1 0.1 \
    'random' &
PID_RANDOM_A=$!

# ─────────────────────────────────────────────────────────────
# Wait for all processes to finish
# ─────────────────────────────────────────────────────────────
# wait ${PID_STORM_C}
# wait ${PID_STORM_A}
# echo "GPU 0 runs complete!"

# wait ${PID_RANDOM_C}
# wait ${PID_RANDOM_A}
# echo "GPU 1 runs complete!"

echo ""
echo "=============================================="
echo "  All SEVIR runs complete. Check wandb."
echo "=============================================="