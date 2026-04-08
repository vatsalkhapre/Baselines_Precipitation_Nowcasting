#!/bin/bash
# ============================================================
# Frequency Multiplier Tuning for Wavelet-Gabor LASTOCast
# 
# Strategy (per level):
#   Phase A: Fix freq_low=0.5, sweep freq_high → find best HF freq
#   Phase B: Fix freq_high=Phase A winner, sweep freq_low → find best LL freq
#
# Levels: J=2, J=3 (separate mode only)
# Wavelet: db4 (run db6 separately if needed)
# Dataset: CIKM
#
# Phase A: 5 runs per level × 2 levels = 10 runs
# Phase B: 5 runs per level × 2 levels = 10 runs (run after Phase A)
# Total: 20 runs
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_conv_less_full_mlp_waveletsgabor2"
SEED=0
WAVE="db4"

# Fixed params
WS_LOW=1.0; A_LOW=1.0; B_LOW=1.0
WS_HIGH=1.0; A_HIGH=1.0; B_HIGH=1.0

# Dataset
DATASET="cikm_latent_32"
SEQ_LEN=15
FRAMES_IN=5
FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="cikm_latent_32_freq_tuning"

run_experiment() {
    local GPU=$1
    local LEVEL=$2
    local F_LOW=$3
    local F_HIGH=$4
    local PHASE=$5

    local TAG="${WAVE}_J${LEVEL}_sep_${PHASE}_flow${F_LOW}_fhigh${F_HIGH}"

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
        --hf_mode separate \
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
        --hf_mode separate \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  ${TAG} complete."
    echo ""
}

# ============================================================
# PHASE A: Fix freq_low=0.5, sweep freq_high
# GPU 0 → J=2 | GPU 1 → J=3
# ============================================================

# echo "=============================================="
# echo "  PHASE A: Sweeping freq_high (freq_low=0.5)"
# echo "  GPU 0 → J=2 | GPU 1 → J=3"
# echo "=============================================="

# run_phase_a_gpu0() {
#     for F_HIGH in 0.5 1.0 1.5 2.0 3.0
#     do
#         run_experiment 0 2 0.5 ${F_HIGH} "phaseA"
#     done
# }

# run_phase_a_gpu1() {
#     for F_HIGH in 0.5 1.0 1.5 2.0 3.0
#     do
#         run_experiment 1 3 0.5 ${F_HIGH} "phaseA"
#     done
# }

# run_phase_a_gpu0 &
# PID_A0=$!

# run_phase_a_gpu1 &
# PID_A1=$!

# wait ${PID_A0}
# echo "Phase A GPU 0 (J=2) complete!"

# wait ${PID_A1}
# echo "Phase A GPU 1 (J=3) complete!"

# echo ""
# echo "=============================================="
# echo "  PHASE A COMPLETE"
# echo ""
# echo "  CHECK RESULTS NOW!"
# echo "  Find the best freq_high for J=2 and J=3"
# echo "  Then update BEST_F_HIGH_J2 and BEST_F_HIGH_J3 below"
# echo "  and uncomment Phase B to run it."
# echo "=============================================="

# ============================================================
# PHASE B: Fix freq_high=BEST, sweep freq_low
# UNCOMMENT AFTER CHECKING PHASE A RESULTS
# Update the BEST values below based on Phase A results
# ============================================================

BEST_F_HIGH_J2=1.5   # <-- UPDATE with Phase A winner for J=2
BEST_F_HIGH_J2_2=1.0   # <-- UPDATE with Phase A winner for J=3

# echo "=============================================="
# echo "  PHASE B: Sweeping freq_low"
# echo "  J=2 best freq_high=${BEST_F_HIGH_J2}"
# echo "  J=3 best freq_high=${BEST_F_HIGH_J3}"
# echo "=============================================="

run_phase_b_gpu0() {
    for W_S in 0.1 0.25 0.75 0.5 1.0 1.25 1.5 1.75 2.0 2.5 3.0 4.0
    do
        run_experiment 0 2 ${F_LOW} ${BEST_F_HIGH_J2} "phaseB"
    done
}

run_phase_b_gpu1() {
    for F_LOW in 0.1 0.25 0.75 0.5 1.0
    do
        run_experiment 1 2 ${F_LOW} ${BEST_F_HIGH_J2_2} "phaseB"
    done
}

run_phase_b_gpu0 &
PID_B0=$!

run_phase_b_gpu1 &
PID_B1=$!

wait ${PID_B0}
echo "Phase B GPU 0 (J=2) complete!"

wait ${PID_B1}
echo "Phase B GPU 1 (J=3) complete!"

echo ""
echo "=============================================="
echo "  PHASE B COMPLETE — All tuning done!"
echo "=============================================="
