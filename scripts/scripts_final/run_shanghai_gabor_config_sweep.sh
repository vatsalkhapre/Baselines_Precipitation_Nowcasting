#!/bin/bash
# ============================================================
# Gabor Config Sweep — Shanghai (3 GPUs)
# Config A (near-MLP):    beta=100, freq_multiplier=0.1
# Config B (current):     beta=YOUR_DEFAULT, freq_multiplier=YOUR_DEFAULT
# Config C (strong Gabor):beta=0.17, freq_multiplier=4.0
# 1 config per GPU, all 3 in parallel
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# ── Fixed Shanghai best config — UPDATE VALUES ────────────────
WAVE="db6";   LEVEL=3;   HF_MODE="separate"
BLOCKS=4;     FACTOR=3;  K=3;   SPARSITY=0.01
WS_LOW=0.1;   WS_HIGH=1.0
A_LOW=1.0;    A_HIGH=1.0
EPOCHS=50

SHANGHAI_CFG="shanghai_lr_latent_32|25|5|20|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth|gabor_config_sweep_shanghai"

# ── 3 configs ─────────────────────────────────────────────────
# Config A: near-MLP
A_BETA_LOW=1.0;   A_BETA_HIGH=0.17;   A_FREQ_LOW=0.1;  A_FREQ_HIGH=0.1

# Config B: current default — UPDATE THESE
B_BETA_LOW=1.0;   B_BETA_HIGH=1.0;   B_FREQ_LOW=2.0;  B_FREQ_HIGH=0.75

# Config C: strong Gabor
C_BETA_LOW=0.17;  C_BETA_HIGH=0.17;  C_FREQ_LOW=4.0;  C_FREQ_HIGH=4.0

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local CFG_NAME=$2
    local BETA_LOW=$3
    local BETA_HIGH=$4
    local FREQ_LOW=$5
    local FREQ_HIGH=$6

    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${SHANGHAI_CFG}"

    local TAG="config${CFG_NAME}_beta${BETA_LOW}_freq${FREQ_LOW}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | Config ${CFG_NAME}"
    echo "  beta=${BETA_LOW} freq_multiplier=${FREQ_LOW}"
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
        --beta_low ${BETA_LOW} \
        --freq_multiplier_low ${FREQ_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${BETA_HIGH} \
        --freq_multiplier_high ${FREQ_HIGH} \
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
        --beta_low ${BETA_LOW} \
        --freq_multiplier_low ${FREQ_LOW} \
        --weight_scale_high ${WS_HIGH} \
        --alpha_high ${A_HIGH} \
        --beta_high ${BETA_HIGH} \
        --freq_multiplier_high ${FREQ_HIGH} \
        --wave ${WAVE} \
        --wavelet_level ${LEVEL} \
        --hf_mode ${HF_MODE} \
        --afno_blocks ${BLOCKS} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  Done: Config ${CFG_NAME}"
    echo ""
}

echo "=============================================="
echo "  Gabor Config Sweep — Shanghai"
echo "  GPU 0 → Config A (near-MLP)"
echo "  GPU 1 → Config B (current default)"
echo "  GPU 2 → Config C (strong Gabor)"
echo "=============================================="
echo ""

run_experiment 0 A ${A_BETA_LOW} ${A_BETA_HIGH} ${A_FREQ_LOW} ${A_FREQ_HIGH} &
PID_GPU0=$!

run_experiment 1 B ${B_BETA_LOW} ${B_BETA_HIGH} ${B_FREQ_LOW} ${B_FREQ_HIGH} &
PID_GPU1=$!

run_experiment 2 C ${C_BETA_LOW} ${C_BETA_HIGH} ${C_FREQ_LOW} ${C_FREQ_HIGH} &
PID_GPU2=$!

wait ${PID_GPU0}
echo "GPU 0 (Config A) complete!"

wait ${PID_GPU1}
echo "GPU 1 (Config B) complete!"

wait ${PID_GPU2}
echo "GPU 2 (Config C) complete!"

echo ""
echo "=============================================="
echo "  Shanghai gabor config sweep complete."
echo "=============================================="
