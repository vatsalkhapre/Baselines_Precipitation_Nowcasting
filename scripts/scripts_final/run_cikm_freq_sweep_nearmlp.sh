#!/bin/bash
# ============================================================
# CIKM — freq sweep (near-MLP regime)
# Fixed: alpha=1.0, beta=100 (Config A winner)
# Sweep: 3 valid combos keeping F_low < F_high, both very low
#   Combo 1: freq_low=0.05, freq_high=0.1
#   Combo 2: freq_low=0.05, freq_high=0.2
#   Combo 3: freq_low=0.1,  freq_high=0.2
#
# GPU 0 → Combo 1, Combo 3 (sequential)
# GPU 1 → Combo 2
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Fixed CIKM best config ────────────────────────────────────
WAVE="db4";   LEVEL=2;   HF_MODE="separate"
BLOCKS=1;     FACTOR=1;  K=7;   SPARSITY=0.01
WS_LOW=0.1;   WS_HIGH=0.25
A_LOW=1.0;    A_HIGH=1.0
B_LOW=100;    B_HIGH=100      # Config A winner
EPOCHS=50

CIKM_CFG="cikm_latent_32|15|5|10|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth|freq_sweep_cikm_nearmlp"

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local F_LOW=$2
    local F_HIGH=$3

    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${CIKM_CFG}"

    local TAG="flow${F_LOW}_fhigh${F_HIGH}_b${B_LOW}_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | freq_low=${F_LOW} freq_high=${F_HIGH}"
    echo "=============================================="

    # ── Train ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
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
        --afno_blocks ${BLOCKS} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  Done: ${DS_SHORT} | freq_low=${F_LOW} freq_high=${F_HIGH}"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# GPU 0 → Combo 1 (0.05, 0.1) then Combo 3 (0.1, 0.2)
# GPU 1 → Combo 2 (0.05, 0.2)
# ─────────────────────────────────────────────────────────────

run_gpu0() {
    run_experiment 0 0.2 0.5
    run_experiment 0 0.1 0.5
    run_experiment 0 0.3 0.5
    run_experiment 0 0.05 0.5
}

run_gpu1() {
    run_experiment 1 0.1 1.0
    run_experiment 1 0.2 1.0
    run_experiment 1 0.4 1.0
    run_experiment 0 0.05 1.0
}

echo "=============================================="
echo "  CIKM freq sweep — near-MLP regime"
echo "  GPU 0 → (flow=0.05, fhigh=0.1) + (flow=0.1, fhigh=0.2)"
echo "  GPU 1 → (flow=0.05, fhigh=0.2)"
echo "  Fixed: beta=${B_LOW}, F_low < F_high enforced"
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
echo "  CIKM freq sweep complete. Check wandb."
echo "  Next: fix best (freq_low, freq_high) combo."
echo "=============================================="
