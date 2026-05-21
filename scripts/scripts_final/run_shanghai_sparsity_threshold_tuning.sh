#!/bin/bash
# ============================================================
# Sparsity Threshold Tuning: 3 values per dataset
# Each dataset has its own best AFNO + wavelet config
# Values: [0.001, 0.01, 0.1]
#
# GPU 0 → SHANGHAI (3 runs) then SHANGHAI (3 runs)
# GPU 1 → Shanghai (3 runs)
# ============================================================
 
BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0
 
# ── Sparsity values to sweep ──────────────────────────────────
SPARSITY_VALUES=(0 0.001 0.01 0.1)
 
# ── UPDATE: SHANGHAI best config ──────────────────────────────
SHANGHAI_WAVE="db6";      SHANGHAI_LEVEL=3;  SHANGHAI_HF_MODE="separate"
SHANGHAI_BLOCKS=4;        SHANGHAI_FACTOR=3; SHANGHAI_K=9
SHANGHAI_WS_LOW=0.1;  SHANGHAI_A_LOW=1.0;  SHANGHAI_B_LOW=1.0;  SHANGHAI_F_LOW=2.0
SHANGHAI_WS_HIGH=1.0; SHANGHAI_A_HIGH=1.0; SHANGHAI_B_HIGH=1.0; SHANGHAI_F_HIGH=0.75
SHANGHAI_CFG="shanghai_lr_latent_32|25|5|20|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth|sparsity_tuning_SHANGHAI"

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local SPARSITY=$2
    local DATASET_CFG=$3
    local WAVE=$4
    local LEVEL=$5
    local HF_MODE=$6
    local BLOCKS=$7
    local FACTOR=$8
    local K=$9
    local WS_LOW=${10}; local A_LOW=${11};  local B_LOW=${12};  local F_LOW=${13}
    local WS_HIGH=${14}; local A_HIGH=${15}; local B_HIGH=${16}; local F_HIGH=${17}
 
    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${DATASET_CFG}"
 
    local TAG="sparsity${SPARSITY}_b${BLOCKS}_f${FACTOR}_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)
 
    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | sparsity=${SPARSITY}"
    echo "=============================================="
 
    # ── Train ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent_model_novelty.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${TAG}" \
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
 
    echo "  Done: ${DS_SHORT} | sparsity=${SPARSITY}"
    echo ""
}


run_gpu0() {
    echo "=== GPU 0: CIKM ==="
    
    run_experiment 0 0.001 \
        "${SHANGHAI_CFG}" ${SHANGHAI_WAVE} ${SHANGHAI_LEVEL} ${SHANGHAI_HF_MODE} \
        ${SHANGHAI_BLOCKS} ${SHANGHAI_FACTOR} ${SHANGHAI_K} \
        ${SHANGHAI_WS_LOW} ${SHANGHAI_A_LOW} ${SHANGHAI_B_LOW} ${SHANGHAI_F_LOW} \
        ${SHANGHAI_WS_HIGH} ${SHANGHAI_A_HIGH} ${SHANGHAI_B_HIGH} ${SHANGHAI_F_HIGH}
 

    run_experiment 0 0.01 \
        "${SHANGHAI_CFG}" ${SHANGHAI_WAVE} ${SHANGHAI_LEVEL} ${SHANGHAI_HF_MODE} \
        ${SHANGHAI_BLOCKS} ${SHANGHAI_FACTOR} ${SHANGHAI_K} \
        ${SHANGHAI_WS_LOW} ${SHANGHAI_A_LOW} ${SHANGHAI_B_LOW} ${SHANGHAI_F_LOW} \
        ${SHANGHAI_WS_HIGH} ${SHANGHAI_A_HIGH} ${SHANGHAI_B_HIGH} ${SHANGHAI_F_HIGH}
   
}


run_gpu1() {
    echo "=== GPU 0: CIKM ==="
    
    run_experiment 1 0 \
        "${SHANGHAI_CFG}" ${SHANGHAI_WAVE} ${SHANGHAI_LEVEL} ${SHANGHAI_HF_MODE} \
        ${SHANGHAI_BLOCKS} ${SHANGHAI_FACTOR} ${SHANGHAI_K} \
        ${SHANGHAI_WS_LOW} ${SHANGHAI_A_LOW} ${SHANGHAI_B_LOW} ${SHANGHAI_F_LOW} \
        ${SHANGHAI_WS_HIGH} ${SHANGHAI_A_HIGH} ${SHANGHAI_B_HIGH} ${SHANGHAI_F_HIGH}
 
}

run_gpu2() {
    run_experiment 2 0.1 \
        "${SHANGHAI_CFG}" ${SHANGHAI_WAVE} ${SHANGHAI_LEVEL} ${SHANGHAI_HF_MODE} \
        ${SHANGHAI_BLOCKS} ${SHANGHAI_FACTOR} ${SHANGHAI_K} \
        ${SHANGHAI_WS_LOW} ${SHANGHAI_A_LOW} ${SHANGHAI_B_LOW} ${SHANGHAI_F_LOW} \
        ${SHANGHAI_WS_HIGH} ${SHANGHAI_A_HIGH} ${SHANGHAI_B_HIGH} ${SHANGHAI_F_HIGH}
}


echo "=============================================="
echo "  Sparsity Tuning — 3 values × 3 datasets"
echo "  GPU 0 → SHANGHAI"
echo "  GPU 1 → SHANGHAI"
echo "  GPU 0 → SHANGHAI"
echo "  Sweeping: ${SPARSITY_VALUES[*]}"
echo "=============================================="
echo ""


run_gpu0 &
PID_GPU0=$!
 
run_gpu1 &
PID_GPU1=$!

run_gpu2 &
PID_GPU2=$!

wait ${PID_GPU0}
echo "GPU 0 (SHANGHAI ) complete!"
 
wait ${PID_GPU1}
echo "GPU 1 (SHANGHAI) complete!"

wait ${PID_GPU2}
echo "GPU 2 (SHANGHAI) complete!"

echo ""
echo "=============================================="
echo "  Sparsity tuning complete. Check wandb."
echo "  Next: fix best sparsity per dataset,"
echo "  then tune Gabor params."
echo "=============================================="