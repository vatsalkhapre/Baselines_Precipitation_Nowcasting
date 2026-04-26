#!/bin/bash
# ============================================================
# Beta Tuning — CIKM
# Sweeping beta_low + beta_high: [1.0, 10, 100]
# Fixed: alpha=1.0, freq_multiplier=1.0 (both low and high)
# All other params: best from sparsity tuning
#
# 3 values → GPU 0 (beta=1.0, beta=10), GPU 1 (beta=100)
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Fixed: best CIKM config ───────────────────────────────────
WAVE="db4"; LEVEL=2; HF_MODE="separate"
BLOCKS=1; FACTOR=1; K=7; SPARSITY=0.01
WS_LOW=0.1;  WS_HIGH=0.25
A_LOW=1.0;   A_HIGH=1.0
F_LOW=1.0;   F_HIGH=1.0    # freq_multiplier fixed at 1.0

CIKM_CFG="cikm_latent_32|15|5|10|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth|beta_tuning_cikm"

# ── Beta values to sweep ──────────────────────────────────────
BETA_VALUES=(1.0 10 100)

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local BETA=$2

    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${CIKM_CFG}"

    local TAG="beta${BETA}_${WAVE}_J${LEVEL}_${HF_MODE}_b${BLOCKS}_f${FACTOR}_sp${SPARSITY}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | beta=${BETA}"
    echo "=============================================="

    # ── Train ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${TAG}" \
        --epochs 50 \
        --ae_ckpt_path "${AE_CKPT}" \
        --valid \
        --seq_len ${SEQ_LEN} \
        --seed ${SEED} \
        --frames_in ${FRAMES_IN} \
        --frames_out ${FRAMES_OUT} \
        --weight_scale_low ${WS_LOW} \
        --alpha_low ${A_LOW} \
        --beta_low ${BETA} \
        --freq_multiplier_low ${F_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${BETA} \
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
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

    # ── Eval ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
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
        --beta_low ${BETA} \
        --freq_multiplier_low ${F_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${BETA} \
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

    echo "  Done: ${DS_SHORT} | beta=${BETA}"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# GPU 0 → beta=1.0, beta=10  (sequential)
# GPU 1 → beta=100
# ─────────────────────────────────────────────────────────────

run_gpu0() {
    run_experiment 0 1.0
    run_experiment 0 10
}

run_gpu1() {
    run_experiment 1 100
}

echo "=============================================="
echo "  Beta Tuning — CIKM"
echo "  GPU 0 → beta=1.0, beta=10"
echo "  GPU 1 → beta=100"
echo "  Fixed: alpha=1.0, freq_multiplier=1.0"
echo "=============================================="
echo ""

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

wait ${PID_GPU0}
echo "GPU 0 complete!"

wait ${PID_GPU1}
echo "GPU 1 complete!"

echo ""
echo "=============================================="
echo "  CIKM beta tuning complete. Check wandb."
echo "=============================================="
