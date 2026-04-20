#!/bin/bash
# ============================================================
# Weight Scale Tuning for Wavelet-Gabor LASTOCast
# 
# Config: J=2, db4, separate mode
# Locked: freq_low=0.75, freq_high=1.0
#
# Phase A: Fix ws_low=1.0, sweep ws_high (12 values)
#   GPU 0 → first 6 values | GPU 1 → last 6 values
#
# Phase B: Fix ws_high=Phase A winner, sweep ws_low (12 values)
#   UNCOMMENT AFTER PHASE A
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_waveletsgabor2"
SEED=0
WAVE="db4"
LEVEL=2
HF_MODE="separate"

# Locked freq params (from tuning)
F_LOW=2.0
F_HIGH=0.75

# Fixed Gabor params
A_LOW=1.0; B_LOW=1.0
A_HIGH=1.0; B_HIGH=1.0

# Dataset
DATASET="meteo_lr_latent_32"
SEQ_LEN=25
FRAMES_IN=5
FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
EXP_DIR="Meteonet_wavelet_variant"

run_experiment() {
    local GPU=$1
    local WS_LOW=$2
    local WS_HIGH=$3
    local PHASE=$4

    local TAG="${WAVE}_J${LEVEL}_sep_${PHASE}_wslow${WS_LOW}_wshigh${WS_HIGH}"

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
        --hf_mode ${HF_MODE} \
        --num_workers 8 \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name "${BACKBONE}_meteonet_${TAG}"

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
        --hf_mode ${HF_MODE} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  ${TAG} complete."
    echo ""
}

# ============================================================
# PHASE A: Fix ws_low=1.0, sweep ws_high
# ============================================================

# echo "=============================================="
# echo "  PHASE A: Sweeping ws_high (ws_low=1.0)"
# echo "  GPU 0 → 0.1 0.25 0.5 0.75 1.0 1.25"
# echo "  GPU 1 → 1.5 1.75 2.0 3.0 4.0"
# echo "=============================================="

# run_phase_a_gpu0() {
#     for WS_HIGH in 0.1 0.25 0.5 0.75 1.25
#     do
#         run_experiment 0 1.0 ${WS_HIGH} "phaseA"
#     done
# }

# run_phase_a_gpu1() {
#     for WS_HIGH in 1.5 1.75 2.0 3.0 
#     do
#         run_experiment 1 1.0 ${WS_HIGH} "phaseA"
#     done
# }

# run_phase_a_gpu0 &
# PID_A0=$!

# run_phase_a_gpu1 &
# PID_A1=$!

# wait ${PID_A0}
# echo "Phase A GPU 0 complete!"

# wait ${PID_A1}
# echo "Phase A GPU 1 complete!"

# echo ""
# echo "=============================================="
# echo "  PHASE A COMPLETE"
# echo ""
# echo "  CHECK RESULTS: Find best ws_high"
# echo "  Update BEST_WS_HIGH below and uncomment Phase B"
# echo "=============================================="

# ============================================================
# PHASE B: Fix ws_high=BEST, sweep ws_low
# UNCOMMENT AFTER PHASE A
# ============================================================

BEST_WS_HIGH=1.0   # <-- UPDATE with Phase A winner

echo "=============================================="
echo "  PHASE B: Sweeping ws_low (ws_high=${BEST_WS_HIGH})"
echo "=============================================="

run_phase_b_gpu1() {
    for WS_LOW in 0.1 0.25 0.5 0.75 1.25
    do
        run_experiment 1 ${WS_LOW} ${BEST_WS_HIGH} "phaseB"
    done
}


run_phase_a_gpu0() {
    for WS_LOW in 1.5 1.75 2.0 3.0 
    do
        run_experiment 0 ${WS_LOW} ${BEST_WS_HIGH} "phaseA"
    done
}

run_phase_b_gpu0 &
PID_B0=$!

run_phase_b_gpu1 &
PID_B1=$!

wait ${PID_B0}
echo "Phase B GPU 0 complete!"

wait ${PID_B1}
echo "Phase B GPU 1 complete!"

echo ""
echo "=============================================="
echo "  PHASE B COMPLETE — Weight scale tuning done!"
echo "  Best config: ws_low=?, ws_high=${BEST_WS_HIGH}"
echo "=============================================="
