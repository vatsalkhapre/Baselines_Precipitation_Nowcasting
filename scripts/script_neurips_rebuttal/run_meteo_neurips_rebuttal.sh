#!/bin/bash
# ============================================================
# NeurIPS rebuttal ablation ladder — Meteonet
# 6 DAWN-Cast ablations (frozen-gamma "expgabor"), split across 2 GPUs.
#   ab1 MLP only            (MSE)
#   ab2 Wavelet + MLP       (MSE)
#   ab3 Wavelet+Gabor+MLP   (MSE, no SRST)
#   ab4 + 1 SRSTResBlock+STR(MSE)
#   ab5 + 2 SRSTResBlock+STR(MSE)  <- full model
#   ab6 full + FACL         (FACL)
# GPU0: ab1, ab3, ab5   |   GPU1: ab2, ab4, ab6   (run concurrently)
# Hyperparameters mirror scripts/scripts_final/run_meteo_nosrst_mse.sh
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"

DATASET="meteo_lr_latent_32"
SEQ_LEN=25
FRAMES_IN=5
FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
EXP_DIR="neurips_rebuttal_meteo"
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

# Gabor frequencies (used by ab3-ab6; ignored by ab1/ab2)
F_LOW=1.09
F_HIGH=1.12

run_experiment() {
    local GPU=$1
    local BACKBONE=$2

    local TAG="Meteonet_${BACKBONE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "GPU ${GPU} | ${BACKBONE} | flow=${F_LOW} fhigh=${F_HIGH}"
    echo "=============================================="

    # ---- Train ----
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
        --wandb_project_name DAWNCAST_neurips_rebuttal \
        --run_name "${DS_SHORT}_${BACKBONE}"

    # ---- Eval ----
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

GPU0_TASKS=(
    "dawncast_ab1_mlp_only"
    "dawncast_ab3_wavelet_mlp_gabor"
    "dawncast_ab5_full"
)
GPU1_TASKS=(
    "dawncast_ab2_wavelet_mlp"
    "dawncast_ab4_srst1"
    "dawncast_ab6_full_facl"
)

run_gpu0() { for b in "${GPU0_TASKS[@]}"; do run_experiment 0 "$b"; done; }
run_gpu1() { for b in "${GPU1_TASKS[@]}"; do run_experiment 1 "$b"; done; }

echo "=============================================="
echo "GPU0 runs ${#GPU0_TASKS[@]} ablations | GPU1 runs ${#GPU1_TASKS[@]} ablations"
echo "=============================================="

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait $PID0; echo "GPU0 finished."
wait $PID1; echo "GPU1 finished."

echo "All Meteonet NeurIPS-rebuttal ablations finished."
