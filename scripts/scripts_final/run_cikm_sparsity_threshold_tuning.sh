#!/bin/bash
# ============================================================
# Sparsity Threshold Tuning: 3 values per dataset
# Each dataset has its own best AFNO + wavelet config
# Values: [0.001, 0.01, 0.1]
#
# GPU 0 → CIKM (3 runs) then MeteoNet (3 runs)
# GPU 1 → Shanghai (3 runs)
# ============================================================
 
BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0
 
# ── Sparsity values to sweep ──────────────────────────────────
SPARSITY_VALUES=(0 0.001 0.01 0.1)
 
# ── UPDATE: CIKM best config ──────────────────────────────────
CIKM_WAVE="db4";      CIKM_LEVEL=2;  CIKM_HF_MODE="separate"
CIKM_BLOCKS=1;        CIKM_FACTOR=1; CIKM_K=7
CIKM_WS_LOW=0.1;  CIKM_A_LOW=1.0;  CIKM_B_LOW=1.0;  CIKM_F_LOW=0.75
CIKM_WS_HIGH=0.25; CIKM_A_HIGH=1.0; CIKM_B_HIGH=1.0; CIKM_F_HIGH=1.0
CIKM_CFG="cikm_latent_32|15|5|10|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth|sparsity_tuning_cikm"

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

# ─────────────────────────────────────────────────────────────
# GPU 0 → CIKM (3 runs) then MeteoNet (3 runs)
# GPU 1 → Shanghai (3 runs)
# ─────────────────────────────────────────────────────────────

run_gpu0() {
    echo "=== GPU 0: CIKM ==="
    
    run_experiment 0 0.001 \
        "${CIKM_CFG}" ${CIKM_WAVE} ${CIKM_LEVEL} ${CIKM_HF_MODE} \
        ${CIKM_BLOCKS} ${CIKM_FACTOR} ${CIKM_K} \
        ${CIKM_WS_LOW} ${CIKM_A_LOW} ${CIKM_B_LOW} ${CIKM_F_LOW} \
        ${CIKM_WS_HIGH} ${CIKM_A_HIGH} ${CIKM_B_HIGH} ${CIKM_F_HIGH}
 

    run_experiment 0 0.01 \
        "${CIKM_CFG}" ${CIKM_WAVE} ${CIKM_LEVEL} ${CIKM_HF_MODE} \
        ${CIKM_BLOCKS} ${CIKM_FACTOR} ${CIKM_K} \
        ${CIKM_WS_LOW} ${CIKM_A_LOW} ${CIKM_B_LOW} ${CIKM_F_LOW} \
        ${CIKM_WS_HIGH} ${CIKM_A_HIGH} ${CIKM_B_HIGH} ${CIKM_F_HIGH}
   
}


run_gpu1() {
    echo "=== GPU 0: CIKM ==="
    
    run_experiment 1 0 \
        "${CIKM_CFG}" ${CIKM_WAVE} ${CIKM_LEVEL} ${CIKM_HF_MODE} \
        ${CIKM_BLOCKS} ${CIKM_FACTOR} ${CIKM_K} \
        ${CIKM_WS_LOW} ${CIKM_A_LOW} ${CIKM_B_LOW} ${CIKM_F_LOW} \
        ${CIKM_WS_HIGH} ${CIKM_A_HIGH} ${CIKM_B_HIGH} ${CIKM_F_HIGH}
 

    run_experiment 1 0.1 \
        "${CIKM_CFG}" ${CIKM_WAVE} ${CIKM_LEVEL} ${CIKM_HF_MODE} \
        ${CIKM_BLOCKS} ${CIKM_FACTOR} ${CIKM_K} \
        ${CIKM_WS_LOW} ${CIKM_A_LOW} ${CIKM_B_LOW} ${CIKM_F_LOW} \
        ${CIKM_WS_HIGH} ${CIKM_A_HIGH} ${CIKM_B_HIGH} ${CIKM_F_HIGH}
    

}
echo "=============================================="
echo "  Sparsity Tuning — 3 values × 3 datasets"
echo "  GPU 0 → CIKM"
echo "  GPU 1 → CIKM"
echo "  Sweeping: ${SPARSITY_VALUES[*]}"
echo "=============================================="
echo ""


run_gpu0 &
PID_GPU0=$!
 
run_gpu1 &
PID_GPU1=$!
 
wait ${PID_GPU0}
echo "GPU 0 (CIKM ) complete!"
 
wait ${PID_GPU1}
echo "GPU 1 (CIKM) complete!"
 
echo ""
echo "=============================================="
echo "  Sparsity tuning complete. Check wandb."
echo "  Next: fix best sparsity per dataset,"
echo "  then tune Gabor params."
echo "=============================================="