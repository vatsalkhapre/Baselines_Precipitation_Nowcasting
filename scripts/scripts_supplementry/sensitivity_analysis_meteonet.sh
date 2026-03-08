#!/bin/bash
# Sensitivity Analysis: One hyperparameter at a time
# Defaults: dim=64, size_factor=1.0, freq_multiplier=1.0

GPU=1
BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_gabor2"
DATASET="meteo_lr_latent_32"
EXP_DIR="meteo_lr_latent_32_sensitivity_analysis"
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
SEED=0

# Defaults
DEF_DIM=64
DEF_SF=1.0
DEF_F=1.5
DEF_WS=1.5
DEF_A=1.0
DEF_B=1.0

run_experiment() {
    local DIM=$1 SF=$2 F=$3 TAG=$4

    echo "=============================================="
    echo "  ${TAG}: dim=${DIM}, size_factor=${SF}, freq=${F}"
    echo "=============================================="

    # Train
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${BACKBONE}_${TAG}" \
        --epochs 50 \
        --ae_ckpt_path "${AE_CKPT}" \
        --valid \
        --seq_len 25 \
        --seed ${SEED} \
        --frames_in 5 \
        --frames_out 20 \
        --weight_scale ${DEF_WS} \
        --alpha ${DEF_A} \
        --beta ${DEF_B} \
        --freq_multiplier ${F} \
        --hidden_dim ${DIM} \
        --size_factor ${SF} \
        --num_workers 8 \
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name "${BACKBONE}_sensitivity_${TAG}"

    # Eval
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note "${BACKBONE}_${TAG}" \
        --ae_ckpt_path "${AE_CKPT}" \
        --eval \
        --seed ${SEED} \
        --seq_len 25 \
        --frames_in 5 \
        --frames_out 20 \
        --weight_scale ${DEF_WS} \
        --alpha ${DEF_A} \
        --beta ${DEF_B} \
        --freq_multiplier ${F} \
        --hidden_dim ${DIM} \
        --size_factor ${SF} \
        --num_workers 8 \
        --wandb_state 'offline'

    echo "  ${TAG} complete."
    echo ""
}


# ==============================================================
# 2. Vary size_factor (fix dim=64, freq_multiplier=1.0)
# ==============================================================
for SF in 0.25 0.5 1.0 2.0 4.0
do
    run_experiment ${DEF_DIM} ${SF} ${DEF_F} "sf_${SF}"
done

# ==============================================================
# 3. Vary freq_multiplier (fix dim=64, size_factor=1.0)
# ==============================================================
for F in 0.25 0.5 1.0 1.5 2.0
do
    run_experiment ${DEF_DIM} ${DEF_SF} ${F} "freq_${F}"
done

echo "=============================================="
echo "  All sensitivity experiments complete!"
echo "  Total runs: 15 (13 unique + 2 duplicate defaults)"
echo "=============================================="