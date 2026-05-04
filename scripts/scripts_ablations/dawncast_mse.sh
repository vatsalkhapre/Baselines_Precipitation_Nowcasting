#!/bin/bash
# ============================================================
# MSE Variant — CIKM + Shanghai
# GPU 0 → CIKM
# GPU 1 → Shanghai
# Both in parallel
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_mse_final"
RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"
SEED=0

# ── AE checkpoints ────────────────────────────────────────────
AE_CIKM="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
AE_SHANGHAI="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth"

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local DATASET=$2
    local SEQ_LEN=$3; local FRAMES_IN=$4; local FRAMES_OUT=$5
    local AE_CKPT=$6; local EXP_DIR=$7
    local WAVE=$8; local LEVEL=$9; local HF_MODE=${10}
    local BLOCKS=${11}; local FACTOR=${12}; local K=${13}; local SPARSITY=${14}
    local WS_LOW=${15}; local WS_HIGH=${16}
    local A_LOW=${17};  local A_HIGH=${18}
    local B_LOW=${19};  local B_HIGH=${20}
    local F_LOW=${21};  local F_HIGH=${22}

    local TAG="mse_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | MSE variant"
    echo "=============================================="

    # ── Train ──
    # CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
    #     --backbone ${BACKBONE} \
    #     --dataset ${DATASET} \
    #     --exp_dir ${EXP_DIR} \
    #     --exp_note "${TAG}" \
    #     --epochs 50 \
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
    #     --afno2D_hidden_size_factor ${FACTOR} \
    #     --afno_sparsity_threshold ${SPARSITY} \
    #     --conv_kernel ${K} \
    #     --num_workers 8 \
    #     --wandb_state 'online' \
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
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  Done: ${DS_SHORT} | MSE variant"; echo ""
}

echo "=============================================="
echo "  MSE Variant — CIKM + Shanghai (parallel)"
echo "  GPU 0 → CIKM | GPU 1 → Shanghai"
echo "=============================================="
echo ""

# GPU 0 → CIKM: db4 L2, blocks=1, factor=1, k=7
run_experiment 0 \
    cikm_latent_32 15 5 10 \
    ${AE_CIKM} mse_cikm \
    db4 2 separate \
    1 1 7 0.01 \
    0.1 0.25 1.0 1.0 100 100 0.1 0.1 &
PID_GPU0=$!

# GPU 1 → Shanghai: db6 L3, blocks=4, factor=3, k=3
run_experiment 1 \
    shanghai_lr_latent_32 25 5 20 \
    ${AE_SHANGHAI} mse_shanghai \
    db6 3 separate \
    4 3 3 0.01 \
    0.1 1.0 1.0 1.0 0.17 0.17 4.0 4.0 &
PID_GPU1=$!

wait ${PID_GPU0}
echo "GPU 0 (CIKM) complete!"

wait ${PID_GPU1}
echo "GPU 1 (Shanghai) complete!"

echo ""
echo "=============================================="
echo "  MSE variant runs complete. Check wandb."
echo "=============================================="