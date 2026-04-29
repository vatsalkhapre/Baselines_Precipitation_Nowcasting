#!/bin/bash
# ============================================================
# Ablation Study — CIKM
# All 7 ablations run sequentially on GPU 0
# Best CIKM params fixed throughout
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"
SEED=0
GPU=0

# ── Best CIKM params ──────────────────────────────────────────
DATASET="cikm_latent_32"
SEQ_LEN=15; FRAMES_IN=5; FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="ablations_cikm"
EPOCHS=50

WAVE="db4";     LEVEL=2;    HF_MODE="separate"
BLOCKS=1;       FACTOR=1;   K=7;    SPARSITY=0.01
WS_LOW=0.1;     WS_HIGH=0.25
A_LOW=1.0;      A_HIGH=1.0
B_LOW=100;      B_HIGH=100
F_LOW=0.1;      F_HIGH=0.1

# ── Ablation backbone names ───────────────────────────────────
ABL1="amplinet_latent_falfcl_only_incr1_mlp_only"
ABL2="amplinet_latent_falfcl_only_incr2_mlp_gabor"
ABL3="amplinet_latent_falfcl_only_incr3_mlp_gabor_wavelet"
ABL4="amplinet_latent_falfcl_only_incr3p5_mlp_gabor_wavelet_afno_only"
ABL5="amplinet_latent_falfcl_only_incr4_mlp_gabor_wavelet_conv"
# ─────────────────────────────────────────────────────────────
run_experiment() {
    local BACKBONE=$1
    local NOTE=$2

    local TAG="${NOTE}_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | CIKM | ${NOTE}"
    echo "=============================================="

    # ── Train ──
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
        --wandb_state 'online' \
        --wandb_project_name 'Nowcasting_ablations' \
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

    # ── Eval ──
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
        --wandb_state 'offline'

    echo "  Done: CIKM | ${NOTE}"
    echo ""
}

echo "=============================================="
echo "  Ablation Study — CIKM (GPU 0, sequential)"
echo "  7 ablations total"
echo "=============================================="
echo ""

run_experiment ${ABL1}  "inc 1"
run_experiment ${ABL2} "inc 2"
run_experiment ${ABL3} "inc 3"
run_experiment ${ABL4}  "inc 4"
run_experiment ${ABL5}  "inc 5"
echo "=============================================="
echo "  CIKM ablations complete. Check wandb."
echo "=============================================="