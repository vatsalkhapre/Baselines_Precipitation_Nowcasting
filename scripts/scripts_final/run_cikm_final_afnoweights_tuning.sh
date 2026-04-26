#!/bin/bash
# ============================================================
# AFNO Diversity Tuning: 6 param-neutral combos
# Explores block structure vs hidden width tradeoff

# Combos (all param-neutral relative to baseline):
#   (1,1) baseline
#   (2,1) more structure, same capacity
#   (4,1) highly structured, least capacity
#   (2,2) same params as (1,1), richer hidden
#   (4,2) same params as (2,1), richer hidden
#   (4,4) same params as (1,1), max diversity + expressiveness

# GPU 0 → first 3 combos sequentially
# GPU 1 → last 3 combos sequentially
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Best wavelet config from Phase 1 (fixed) ──────────────────
WAVE="db4"
LEVEL=2
HF_MODE="separate"
K_SPATIAL=7

# ── Best Gabor params from previous tuning (fixed) ────────────
WS_LOW=0.1;  A_LOW=1.0;  B_LOW=1.0;  F_LOW=0.75
WS_HIGH=0.25; A_HIGH=1.0; B_HIGH=1.0; F_HIGH=1.0

# ── Fixed AFNO sparsity (tune separately after this sweep) ────
SPARSITY=0.01

# ── Dataset config ─────────────────────────────────────────────
CIKM="cikm_latent_32|15|5|10|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth|afno_diversity_tuning"

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local DATASET_CFG=$2
    local AFNO_BLOCKS=$3
    local AFNO_FACTOR=$4

    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${DATASET_CFG}"

    local TAG="afno_b${AFNO_BLOCKS}_f${AFNO_FACTOR}_${WAVE}_J${LEVEL}_${HF_MODE}_${DATASET}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | blocks=${AFNO_BLOCKS} factor=${AFNO_FACTOR}"
    echo "  TAG: ${TAG}"
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
        --beta_low ${B_LOW} \
        --freq_multiplier_low ${F_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${B_HIGH} \
        --freq_multiplier_high ${F_HIGH} \
        --wave ${WAVE} \
        --wavelet_level ${LEVEL} \
        --hf_mode ${HF_MODE} \
        --afno_blocks ${AFNO_BLOCKS} \
        --afno2D_hidden_size_factor ${AFNO_FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K_SPATIAL} \
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
        --beta_low ${B_LOW} \
        --freq_multiplier_low ${F_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${B_HIGH} \
        --freq_multiplier_high ${F_HIGH} \
        --wave ${WAVE} \
        --wavelet_level ${LEVEL} \
        --hf_mode ${HF_MODE} \
        --afno_blocks ${AFNO_BLOCKS} \
        --afno2D_hidden_size_factor ${AFNO_FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K_SPATIAL} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  Done: blocks=${AFNO_BLOCKS} factor=${AFNO_FACTOR}"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# GPU 0 → (1,1), (2,1), (4,1)  — fixed hidden width, vary blocks
# GPU 1 → (2,2), (4,2), (4,4)  — wider hidden, vary blocks
# ─────────────────────────────────────────────────────────────

run_gpu0() {
    # run_experiment 0 "${CIKM}" 4 3
    # run_experiment 0 "${CIKM}" 2 1
    # run_experiment 0 "${CIKM}" 4 1
    run_experiment 0 "${CIKM}" 1 2
}

# run_gpu1() {
#     # run_experiment 1 "${CIKM}" 4 6
#     # run_experiment 1 "${CIKM}" 4 2
#     # run_experiment 1 "${CIKM}" 4 4
# }

echo "=============================================="
echo "  AFNO Diversity Tuning — 6 combos on CIKM"
echo "  GPU 0 → (1,1) (2,1) (4,1)"
echo "  GPU 1 → (2,2) (4,2) (4,4)"
echo "  Sparsity fixed at ${SPARSITY}"
echo "=============================================="
echo ""

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

wait ${PID_GPU0}
echo "GPU 0 complete: (1,1) (2,1) (4,1)"

wait ${PID_GPU1}
echo "GPU 1 complete: (2,2) (4,2) (4,4)"

echo ""
echo "=============================================="
echo "  All 6 combos done. Check wandb."
echo "  Next: fix best combo, sweep sparsity"
echo "  in [0.001, 0.01, 0.1] on winner only."
echo "=============================================="
