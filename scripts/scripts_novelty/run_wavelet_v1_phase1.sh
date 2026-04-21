#!/bin/bash
# ============================================================
# Phase 1: Wavelet & Level Search for Wavelet-Gabor LASTOCast
# 12 configs × 2 datasets = 24 runs
# Uses GPU 0 and GPU 1 in parallel
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0

# Fixed Gabor params (default best)
WS_LOW=1.0; A_LOW=1.0; B_LOW=1.0; F_LOW=0.5
WS_HIGH=1.0; A_HIGH=1.0; B_HIGH=1.0; F_HIGH=2.0

# Dataset configs
# Format: dataset|seq_len|frames_in|frames_out|ae_ckpt|exp_dir
CIKM="cikm_latent_32|15|5|10|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth|cikm_latent_32_wavelet_search"
SHANGHAI="shanghai_lr_latent_32|25|5|20|/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth|shanghai_lr_latent_32_wavelet_search"

run_experiment() {
    local GPU=$1
    local DATASET_CFG=$2
    local WAVE=$3
    local LEVEL=$4
    local HF_MODE=$5

    # Parse dataset config
    IFS='|' read -r DATASET SEQ_LEN FRAMES_IN FRAMES_OUT AE_CKPT EXP_DIR <<< "${DATASET_CFG}"

    local TAG="${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | ${TAG}"
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
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

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

    echo "  GPU ${GPU} | ${DS_SHORT} | ${TAG} complete."
    echo ""
}

# ============================================================
# Build experiment queue
# 12 configs: 4 wavelets × {J1, J2-shared, J2-separate}
# Run on CIKM only, using both GPUs to split the work
# ============================================================

echo "=============================================="
echo "  Starting Phase 1: Wavelet & Level Search"
echo "  12 configs on CIKM"
echo "  GPU 0 → haar + db2 | GPU 1 → db3 + coif1"
echo "=============================================="
echo ""

# GPU 0: haar and db2
run_gpu0() {
    for WAVE in db4
    do
        # run_experiment 1 "${CIKM}" ${WAVE} 1 shared
        # run_experiment 0 "${CIKM}" ${WAVE} 2 shared
        # run_experiment 0 "${CIKM}" ${WAVE} 2 separate
        # run_experiment 0 "${CIKM}" ${WAVE} 3 separate
        run_experiment 0 "${CIKM}" ${WAVE} 4 separate
    done
}

# GPU 1: db3 and coif1
run_gpu1() {
    for WAVE in db6
    do
        # run_experiment 1 "${CIKM}" ${WAVE} 1 shared
        # run_experiment 1 "${CIKM}" ${WAVE} 2 shared
        # run_experiment 0 "${CIKM}" ${WAVE} 2 separate
        # run_experiment 1 "${CIKM}" ${WAVE} 3 separate
        run_experiment 1 "${CIKM}" ${WAVE} 4 separate
    done
}

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

echo "Waiting for GPU 0 (haar, db2)... PID=${PID_GPU0}"
echo "Waiting for GPU 1 (db3, coif1)... PID=${PID_GPU1}"

wait ${PID_GPU0}
echo "GPU 0 complete!"

wait ${PID_GPU1}
echo "GPU 1 complete!"

echo ""
echo "=============================================="
echo "  Phase 1 complete! All 12 runs finished."
echo "=============================================="
