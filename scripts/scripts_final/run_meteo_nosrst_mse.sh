#!/bin/bash
# ============================================================
# Ablation "+Gabor" (no SRST block) — Meteonet, MSE loss
# Runs 2 models in the "Wavelet + Gabor + MLP" configuration
# (SRST block removed; trained with MSE, no FACL/RandomScheduling):
#   1. expgabor variant : flow=1.09  fhigh=1.12
#   2. gabor    variant : flow=1.09  fhigh=4.41
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"

DATASET="meteo_lr_latent_32"
SEQ_LEN=25
FRAMES_IN=5
FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
EXP_DIR="nosrst_mse_meteo"
EPOCHS=50
SEED=0

WAVE="db6"
LEVEL=1
HF_MODE="separate"

BLOCKS=4
FACTOR=4
K=3
SPARSITY=0.01

WS_LOW=0.1
WS_HIGH=1.0
A_LOW=1.0
A_HIGH=1.0
B_LOW=0.0995
B_HIGH=0.1643

run_experiment() {
    local GPU=$1
    local BACKBONE=$2
    local F_LOW=$3
    local F_HIGH=$4

    local TAG="Meteonet_nosrst_mse_flow${F_LOW}_fhigh${F_HIGH}"
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
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

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

# Model 1: expgabor variant (frozen gamma) — GPU 0
run_experiment 0 \
    "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_expgabor_nosrst_mse_final" \
    1.09 1.12 &

PID1=$!

# Model 2: gabor variant (learnable gamma) — GPU 1
run_experiment 1 \
    "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_nosrst_mse_final" \
    1.09 4.41 &

PID2=$!

# Wait for both experiments (train + eval) to complete
wait $PID1
wait $PID2

echo "All Meteonet no-SRST MSE experiments finished."
