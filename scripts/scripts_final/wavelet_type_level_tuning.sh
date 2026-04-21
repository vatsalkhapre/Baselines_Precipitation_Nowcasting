#!/bin/bash
# ============================================================
# Wavelet Verification: Top configs from Phase 1
# Verifying db4 + db6 at levels 2 and 3 with new backbone
#
# Combos (4 total × shared+separate = 8 runs):
#   db4 J2 shared/separate
#   db4 J3 shared/separate
#   db6 J2 shared/separate
#   db6 J3 shared/separate
#
# GPU 0 → db4 (J2 shared, J2 separate, J3 shared, J3 separate)
# GPU 1 → db6 (J2 shared, J2 separate, J3 shared, J3 separate)
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Fixed AFNO params (use safe defaults for verification) ────
AFNO_BLOCKS=2 
AFNO_FACTOR=2
SPARSITY=0.01
K_SPATIAL=7

# ── Fixed Gabor params from previous best ─────────────────────
WS_LOW=0.1;  A_LOW=1.0;  B_LOW=1.0;  F_LOW=0.75
WS_HIGH=0.25; A_HIGH=1.0; B_HIGH=1.0; F_HIGH=1.0

# ── Dataset config ─────────────────────────────────────────────
CIKM="cikm_latent_32|15|5|10|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth|wavelet_verification"

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local DATASET_CFG=$2
    local WAVE=$3
    local LEVEL=$4
    local HF_MODE=$5

    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${DATASET_CFG}"

    local TAG="${WAVE}_J${LEVEL}_${HF_MODE}_${DATASET_CFG}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${WAVE} J${LEVEL} ${HF_MODE}"
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

    echo "  Done: ${WAVE} J${LEVEL} ${HF_MODE}"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# GPU 0 → db4: J2 shared, J2 separate, J3 shared, J3 separate
# GPU 1 → db6: J2 shared, J2 separate, J3 shared, J3 separate
# ─────────────────────────────────────────────────────────────

run_gpu0() {
    run_experiment 0 "${CIKM}" db4 2 separate
    run_experiment 0 "${CIKM}" db4 3 separate
}

run_gpu1() {
    run_experiment 1 "${CIKM}" db6 2 separate
    run_experiment 1 "${CIKM}" db6 3 separate
}

echo "=============================================="
echo "  Wavelet Verification — 8 runs on CIKM"
echo "  GPU 0 → db4 (J2 sep, J3 sep)"
echo "  GPU 1 → db6 (J2 sep J3 sep)"
echo "  AFNO fixed: blocks=${AFNO_BLOCKS} factor=${AFNO_FACTOR}"
echo "=============================================="
echo ""

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

wait ${PID_GPU0}
echo "GPU 0 complete: db4 done"

wait ${PID_GPU1}
echo "GPU 1 complete: db6 done"

echo ""
echo "=============================================="
echo "  Wavelet verification complete. Check wandb."
echo "  Pick best wave+level+hf_mode, then run:"
echo "  run_afno_diversity_tuning.sh"
echo "=============================================="