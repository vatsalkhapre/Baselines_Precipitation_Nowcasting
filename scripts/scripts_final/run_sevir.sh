#!/bin/bash
# ============================================================
# SEVIR — Config 2 + Config 3 (parallel)
# GPU 1 → Config 2 (strong Gabor, db6 J3)
# GPU 2 → Config 3 (asymmetric freq, db6 J1)
#
# Config 2: W_low:0.1, W_high:1.0, f_low:4.0, f_high:4.0
#           beta:0.17, alpha:1.0, db6, Level:3, k:3, blocks:4, factor:3
#
# Config 3: W_low:0.1, W_high:1.0, f_low:0.1, f_high:4.0
#           beta:0.17, alpha:1.0, db6, Level:1, k:3, blocks:4, factor:4
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novelty.py"
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

    local TAG="${CFG_NAME}_flow${F_LOW}_fhigh${F_HIGH}_b${B_LOW}_${WAVE}_J${LEVEL}_${HF_MODE}"

    echo "=============================================="
    echo "  GPU ${GPU} | SEVIR | ${CFG_NAME}"
    echo "  wave=${WAVE} J=${LEVEL} blocks=${BLOCKS} factor=${FACTOR}"
    echo "  f_low=${F_LOW} f_high=${F_HIGH} beta=${B_LOW}"
    echo "=============================================="

    # ── Train ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${TAG}" \
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
        --wandb_state 'offline' \
        --wandb_project_name 'Alphapre' \
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

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
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  Done: SEVIR | ${CFG_NAME}"
    echo ""
}

echo "=============================================="
echo "  SEVIR — Config 2 + Config 3 (parallel)"
echo "  GPU 1 → Config 2 (db6 J3, strong Gabor)"
echo "  GPU 2 → Config 3 (db6 J1, asymmetric freq)"
echo "=============================================="
echo ""

# GPU 1 → Config 2
run_experiment 0 "config7" "sevir_final_config7" \
    db6 1 separate \
    4 4 3 0.01 \
    0.1 1.0 \
    1.0 1.0 \
    0.17 0.17 \
    4.0 4.0 &
PID_GPU0=$!


run_experiment 1 "config8" "sevir_final_config8" \
    db6 3 separate \
    4 4 3 0.01 \
    0.1 1.0 \
    1.0 1.0 \
    0.17 0.17 \
    0.1 4.0 &
PID_GPU1=$!

# GPU 2 → Config 3
# run_experiment 2 "config6" "sevir_final_config6" \
#     db6 1 separate \
#     4 4 3 0.01 \
#     0.1 1.0 \
#     1.0 1.0 \
#     0.17 0.17 \
#     0.1 3.0 &
# PID_GPU2=$!

wait ${PID_GPU0}
echo "GPU 0 (Config 4) complete!"

wait ${PID_GPU1}
echo "GPU 1 (Config 5) complete!"

# wait ${PID_GPU2}
# echo "GPU 2 (Config 6) complete!"

echo ""
echo "=============================================="
echo "  SEVIR runs complete. Check wandb."
echo "=============================================="