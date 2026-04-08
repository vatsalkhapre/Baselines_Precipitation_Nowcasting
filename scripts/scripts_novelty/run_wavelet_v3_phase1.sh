#!/bin/bash
# ============================================================
# Wavelet LASTOCast V3: Full pipeline per band
# Wavelets: db4, db6 | Levels: J=1, J=2 | HF mode: shared only
# Residual: gabor, mlp, none
# Total: 2 wavelets × 2 levels × 3 residuals = 12 runs on CIKM
# GPU 0 → db4 (6 runs) | GPU 1 → db6 (6 runs)
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_gaborconvwavelets"
SEED=0

# Gabor params (separate for LL and HF)
WS_LOW=1.5; A_LOW=1.0; B_LOW=1.0; F_LOW=0.5
WS_HIGH=1.5; A_HIGH=1.0; B_HIGH=1.0; F_HIGH=2.0

# Dataset
DATASET="cikm_latent_32"
SEQ_LEN=15
FRAMES_IN=5
FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="cikm_latent_32_wavelet_v3_search"

run_experiment() {
    local GPU=$1
    local WAVE=$2
    local LEVEL=$3
    local RES_MODE=$4

    local TAG="${WAVE}_J${LEVEL}_res${RES_MODE}"

    echo "=============================================="
    echo "  GPU ${GPU} | ${TAG}"
    echo "=============================================="

    # Train
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${BACKBONE}_${TAG}" \
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
        --hf_mode shared \
        --residual_mode ${RES_MODE} \
        --num_workers 8 \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name "${BACKBONE}_cikm_${TAG}"

    # Eval
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${BACKBONE}_${TAG}" \
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
        --hf_mode shared \
        --residual_mode ${RES_MODE} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  ${TAG} complete."
    echo ""
}

echo "=============================================="
echo "  Wavelet LASTOCast V3 — Full Pipeline Per Band"
echo "  12 configs on CIKM (shared HF only)"
echo "  GPU 0 → db4 | GPU 1 → db6"
echo "=============================================="
echo ""

# GPU 0: db4
run_gpu0() {
    for LEVEL in 1 2
    do
        for RES_MODE in gabor mlp none
        do
            run_experiment 0 db4 ${LEVEL} ${RES_MODE}
        done
    done
}

# GPU 1: db6
run_gpu1() {
    for LEVEL in 1 2
    do
        for RES_MODE in gabor mlp none
        do
            run_experiment 1 db6 ${LEVEL} ${RES_MODE}
        done
    done
}

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

echo "Waiting for GPU 0 (db4)... PID=${PID_GPU0}"
echo "Waiting for GPU 1 (db6)... PID=${PID_GPU1}"

wait ${PID_GPU0}
echo "GPU 0 (db4) complete!"

wait ${PID_GPU1}
echo "GPU 1 (db6) complete!"

echo ""
echo "=============================================="
echo "  All 12 runs finished!"
echo "=============================================="
