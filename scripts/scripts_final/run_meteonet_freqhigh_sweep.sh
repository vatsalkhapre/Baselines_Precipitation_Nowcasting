#!/bin/bash
# ============================================================
# MeteoNet — freq_multiplier_high sweep (strong Gabor regime)
# Fixed: freq_low=0.1, alpha=1.0, beta=0.17 (Config C winner)
# Sweep: freq_high in [3.0, 4.0, 5.0]
# 1 value per GPU, all 3 in parallel
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Fixed MeteoNet best config ────────────────────────────────
WAVE="db6";   LEVEL=1;   HF_MODE="separate"
BLOCKS=4;     FACTOR=4;  K=3;   SPARSITY=0.01
WS_LOW=0.1;   WS_HIGH=1.0
A_LOW=1.0;    A_HIGH=1.0
B_LOW=0.17;   B_HIGH=0.17     # Config C winner
F_LOW=0.1                      # fixed low for LL band
EPOCHS=50

METEONET_CFG="meteo_lr_latent_32|25|5|20|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth|freq_high_sweep_meteonet"

# ── freq_high values to sweep ─────────────────────────────────
FREQ_HIGH_VALUES=(3.0 4.0 5.0)

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local F_HIGH=$2

    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${METEONET_CFG}"

    local TAG="flow${F_LOW}_fhigh${F_HIGH}_b${B_LOW}_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | freq_high=${F_HIGH}"
    echo "=============================================="

    # ── Train ──
    # CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
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
    #     --afno2D_hidden_size_factor ${FACTOR} \
    #     --afno_sparsity_threshold ${SPARSITY} \
    #     --conv_kernel ${K} \
    #     --num_workers 8 \
    #     --wandb_state 'online' \
    #     --wandb_project_name 'Alphapre' \
    #     --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

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

    echo "  Done: ${DS_SHORT} | freq_high=${F_HIGH}"
    echo ""
}

echo "=============================================="
echo "  MeteoNet freq_high sweep — strong Gabor"
echo "  GPU 0 → freq_high=3.0"
echo "  GPU 1 → freq_high=4.0"
echo "  GPU 2 → freq_high=5.0"
echo "  Fixed: freq_low=${F_LOW}, beta=${B_LOW}"
echo "=============================================="
echo ""

# run_gpu0() {
#     run_experiment 0 1.5
#     run_experiment 0 3.0
# }

# run_gpu1() {
#     run_experiment 0 4.0
# #     # run_experiment 1 2.0
# }

run_gpu2() {
    run_experiment 2 5.0
    run_experiment 2 3.0
}


# run_gpu0 &
# PID_GPU0=$!

# run_gpu1 &
# PID_GPU1=$!

run_gpu2 &
PID_GPU2=$!

# wait ${PID_GPU0}
# echo "GPU 0 (freq_high=3.0) complete!"

wait ${PID_GPU1}
echo "GPU 1 (freq_high=4.0) complete!"

# wait ${PID_GPU2}
# echo "GPU 2 (freq_high=5.0) complete!"

echo ""
echo "=============================================="
echo "  MeteoNet freq_high sweep complete."
echo "  Next: fix best freq_high, verify freq_low"
echo "  in [0.1, 0.25, 0.5]."
echo "=============================================="
