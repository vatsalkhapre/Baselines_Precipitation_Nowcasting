#!/bin/bash
# ============================================================
# Ablation "+Gabor" (no SRST block) — CIKM, MSE loss
# Runs 2 models in the "Wavelet + Gabor + MLP" configuration
# (SRST block removed; trained with MSE, no FACL/RandomScheduling):
#   1. gabor    variant : flow=22.74  fhigh=24.34
#   2. expgabor variant : flow=22.74  fhigh=48.67
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"

DATASET="cikm_latent_32"
SEQ_LEN=15
FRAMES_IN=5
FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="nosrst_mse_cikm"
EPOCHS=50
SEED=0

WAVE="db4"
LEVEL=2
HF_MODE="separate"

BLOCKS=1
FACTOR=1
K=7
SPARSITY=0.01

WS_LOW=0.1
WS_HIGH=0.25
A_LOW=1.0
A_HIGH=1.0
B_LOW=43.1034
B_HIGH=4.8193

run_experiment() {
    local GPU=$1
    local BACKBONE=$2
    local F_LOW=$3
    local F_HIGH=$4

    local TAG="CIKM_nosrst_mse_flow${F_LOW}_fhigh${F_HIGH}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "GPU ${GPU} | ${BACKBONE}"
    echo "flow=${F_LOW} | fhigh=${F_HIGH}"
    echo "=============================================="

    # ---- Train ----
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
        --wandb_state online \
        --wandb_project_name DAWNCAST_nosrst_mse \
        --run_name "CIKM_${BACKBONE}_${DS_SHORT}_${TAG}"

    # ---- Eval ----
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
        --wandb_state offline
}

GPU=1

# Model 1: gabor variant (learnable gamma) — flow=22.74, fhigh=24.34
run_experiment ${GPU} \
    "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_nosrst_mse_final" \
    22.74 24.34

# Model 2: expgabor variant (frozen gamma) — flow=22.74, fhigh=48.67
# run_experiment ${GPU} \
#     "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_expgabor_nosrst_mse_final" \
#     22.74 48.67

echo "All CIKM no-SRST MSE experiments finished."
