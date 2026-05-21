#!/bin/bash
# ============================================================
# Shanghai — Grid Search: freq_low × freq_high (Config A)
# beta=100 (near-MLP regime), F_low < F_high enforced
# 10 valid combos across 3 GPUs
#
# GPU 0 → 3 runs (sequential)
# GPU 1 → 3 runs (sequential)
# GPU 2 → 4 runs (sequential)
# All GPUs in parallel
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Fixed Shanghai best config ────────────────────────────────
WAVE="db6";   LEVEL=3;   HF_MODE="separate"
BLOCKS=4;     FACTOR=3;  K=3;   SPARSITY=0.01
WS_LOW=0.1;   WS_HIGH=1.0
A_LOW=1.0;    A_HIGH=1.0
B_LOW=100;    B_HIGH=100      # Config A
EPOCHS=50

SHANGHAI_CFG="shanghai_lr_latent_32|25|5|20|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth|freq_grid_search_shanghai_configA"

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local F_LOW=$2
    local F_HIGH=$3

    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${SHANGHAI_CFG}"

    local TAG="configA_flow${F_LOW}_fhigh${F_HIGH}_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | flow=${F_LOW} fhigh=${F_HIGH}"
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

    echo "  Done: flow=${F_LOW} fhigh=${F_HIGH}"
    echo ""
}

# ─────────────────────────────────────────────────────────────
# 10 valid combos (F_low < F_high):
# (0.05,0.1) (0.05,0.25) (0.05,0.5) (0.05,1.0)
# (0.1,0.25) (0.1,0.5)  (0.1,1.0)
# (0.25,0.5) (0.25,1.0)
# (0.5,1.0)
#
# GPU 0 → 3 runs: (0.05,0.1)  (0.05,0.25) (0.05,0.5)
# GPU 1 → 3 runs: (0.05,1.0)  (0.1,0.25)  (0.1,0.5)
# GPU 2 → 4 runs: (0.1,1.0)   (0.25,0.5)  (0.25,1.0) (0.5,1.0)
# ─────────────────────────────────────────────────────────────

run_gpu0() {
    run_experiment 0 0.05 0.1
    run_experiment 0 0.05 0.25
    run_experiment 0 0.05 0.5
}

run_gpu1() {
    run_experiment 1 0.05 1.0
    run_experiment 1 0.1  0.25
    run_experiment 1 0.1  0.5
}

run_gpu2() {
    run_experiment 2 0.1  1.0
    run_experiment 2 0.25 0.5
    run_experiment 2 0.25 1.0
    run_experiment 2 0.5  1.0
}

echo "=============================================="
echo "  Shanghai freq grid search — Config A"
echo "  10 combos across 3 GPUs"
echo "  GPU 0 → (0.05,0.1) (0.05,0.25) (0.05,0.5)"
echo "  GPU 1 → (0.05,1.0) (0.1,0.25)  (0.1,0.5)"
echo "  GPU 2 → (0.1,1.0)  (0.25,0.5)  (0.25,1.0) (0.5,1.0)"
echo "  Fixed: beta=100, alpha=1.0"
echo "=============================================="
echo ""

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

run_gpu2 &
PID_GPU2=$!

wait ${PID_GPU0}
echo "GPU 0 complete!"

wait ${PID_GPU1}
echo "GPU 1 complete!"

wait ${PID_GPU2}
echo "GPU 2 complete!"

echo ""
echo "=============================================="
echo "  Shanghai freq grid search complete."
echo "  Check wandb for best (freq_low, freq_high)."
echo "=============================================="
