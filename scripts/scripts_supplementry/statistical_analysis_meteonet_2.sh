#!/bin/bash
# Statistical Analysis: 5-seed runs for LASTOCast on meteo
# Seeds chosen to be well-separated to avoid correlation

SEEDS=(512 1024)
GPU=1
BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_gabor2"
DATASET="meteo_lr_latent_32"
EXP_DIR="meteo_lr_latent_32_statistical_analysis"
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
WEIGHT_SCALE=1.0
ALPHA=1.0
BETA=1.0
FREQ=1.5

for SEED in "${SEEDS[@]}"
do
    echo "=============================================="
    echo "  TRAINING with seed=${SEED}"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${BACKBONE}_${WEIGHT_SCALE}_${ALPHA}_${BETA}_${FREQ}_seed${SEED}" \
        --epochs 50 \
        --ae_ckpt_path "${AE_CKPT}" \
        --valid \
        --seq_len 25 \
        --seed ${SEED} \
        --frames_in 5 \
        --frames_out 20 \
        --weight_scale ${WEIGHT_SCALE} \
        --alpha ${ALPHA} \
        --beta ${BETA} \
        --freq_multiplier ${FREQ} \
        --num_workers 8 \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name "${BACKBONE}_meteonet_seed${SEED}"

    echo "=============================================="
    echo "  EVALUATING with seed=${SEED}"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${BACKBONE}_${WEIGHT_SCALE}_${ALPHA}_${BETA}_${FREQ}_seed${SEED}" \
        --ae_ckpt_path "${AE_CKPT}" \
        --eval \
        --seed ${SEED} \
        --seq_len 25 \
        --frames_in 5 \
        --frames_out 20 \
        --weight_scale ${WEIGHT_SCALE} \
        --alpha ${ALPHA} \
        --beta ${BETA} \
        --freq_multiplier ${FREQ} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo ""
    echo "  Seed ${SEED} complete."
    echo ""
done

echo "=============================================="
echo "  All 5 seeds complete!"
echo "=============================================="