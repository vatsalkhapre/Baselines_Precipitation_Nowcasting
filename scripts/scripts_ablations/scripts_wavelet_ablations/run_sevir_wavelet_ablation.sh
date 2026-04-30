#!/bin/bash
# ============================================================
# Wavelet Ablation — SEVIR
# Best: db6 L2 (skipped)
# 5 runs:
#   Best wavelet (db6) x other levels: L1, L3, L4
#   Best level (L2) x other wavelets:  db4, haar
# Server: .66 GPU → GPU 0, sequential
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"
SEED=0
GPU=0

DATASET="sevir_lr_latent_32"
SEQ_LEN=25; FRAMES_IN=5; FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"
EXP_DIR="wavelet_ablation_sevir"
EPOCHS=50; HF_MODE="separate"
BLOCKS=4; FACTOR=4; K=3; SPARSITY=0.01
WS_LOW=0.1;  WS_HIGH=1.0
A_LOW=1.0;   A_HIGH=1.0
B_LOW=0.17;  B_HIGH=0.17
F_LOW=0.1;   F_HIGH=4.0

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1; local WAVE=$2; local LEVEL=$3
    local TAG="wave${WAVE}_L${LEVEL}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | SEVIR | wave=${WAVE} L=${LEVEL}"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
        --backbone ${BACKBONE} --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} --exp_note "${TAG}" \
        --epochs ${EPOCHS} --ae_ckpt_path "${AE_CKPT}" \
        --valid --seq_len ${SEQ_LEN} --seed ${SEED} \
        --frames_in ${FRAMES_IN} --frames_out ${FRAMES_OUT} \
        --weight_scale_low ${WS_LOW} --alpha_low ${A_LOW} \
        --beta_low ${B_LOW} --freq_multiplier_low ${F_LOW} \
        --weight_scale_high ${WS_HIGH} --alpha_high ${A_HIGH} \
        --beta_high ${B_HIGH} --freq_multiplier_high ${F_HIGH} \
        --wave ${WAVE} --wavelet_level ${LEVEL} \
        --hf_mode ${HF_MODE} --afno_blocks ${BLOCKS} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} --num_workers 8 \
        --wandb_state 'online' --wandb_project_name 'Nowcasting_ablations' \
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

    # CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
    #     --backbone ${BACKBONE} --dataset ${DATASET} \
    #     --exp_dir ${EXP_DIR} --exp_note "${TAG}" \
    #     --ae_ckpt_path "${AE_CKPT}" --eval --seed ${SEED} \
    #     --seq_len ${SEQ_LEN} --frames_in ${FRAMES_IN} \
    #     --frames_out ${FRAMES_OUT} \
    #     --weight_scale_low ${WS_LOW} --alpha_low ${A_LOW} \
    #     --beta_low ${B_LOW} --freq_multiplier_low ${F_LOW} \
    #     --weight_scale_high ${WS_HIGH} --alpha_high ${A_HIGH} \
    #     --beta_high ${B_HIGH} --freq_multiplier_high ${F_HIGH} \
    #     --wave ${WAVE} --wavelet_level ${LEVEL} \
    #     --hf_mode ${HF_MODE} --afno_blocks ${BLOCKS} \
    #     --afno2D_hidden_size_factor ${FACTOR} \
    #     --afno_sparsity_threshold ${SPARSITY} \
    #     --conv_kernel ${K} --num_workers 8 \
    #     --wandb_state 'offline'

    echo "  Done: SEVIR | wave=${WAVE} L=${LEVEL}"; echo ""
}

echo "=============================================="
echo "  SEVIR Wavelet Ablation — 5 runs (GPU 0)"
echo "  Best: db6 L2 (skipped)"
echo "  db6 x [L1,L3,L4] | L2 x [db4,haar]"
echo "=============================================="
echo ""

# Best wavelet (db6) x other levels
run_experiment ${GPU} db6  1
run_experiment ${GPU} db6  3
run_experiment ${GPU} db6  4
# Best level (L2) x other wavelets
run_experiment ${GPU} db4  2
run_experiment ${GPU} haar 2

echo "=============================================="
echo "  SEVIR wavelet ablation complete. Check wandb."
echo "=============================================="
