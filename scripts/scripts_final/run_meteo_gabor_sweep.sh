#!/bin/bash
# ============================================================
# Multi-Combination Runs — Meteonet
# Runs all 20 off-diagonal (flow, fhigh) combinations
# GPU 0 and GPU 1 execute sequential jobs in parallel.
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"

DATASET="meteo_lr_latent_32"
SEQ_LEN=25
FRAMES_IN=5
FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
EXP_DIR="gabor_exp_meteo"
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
    local F_LOW=$2
    local F_HIGH=$3

    local TAG="Meteonet_flow${F_LOW}_fhigh${F_HIGH}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "GPU ${GPU} | flow=${F_LOW} | fhigh=${F_HIGH}"
    echo "=============================================="

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
        --wandb_project_name DAWNCAST_Gabor_sweep \
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

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

FLOWS=(1.09 3.28 8.74 17.49 34.34)
FHIGHS=(0.14 0.42 1.12 2.25 4.41)

echo "Starting Meteonet frequency sweep on GPU 1..."

for F_LOW in "${FLOWS[@]}"; do
    for F_HIGH in "${FHIGHS[@]}"; do
        run_experiment 1 "$F_LOW" "$F_HIGH"
    done
done

echo "All Meteonet experiments finished."