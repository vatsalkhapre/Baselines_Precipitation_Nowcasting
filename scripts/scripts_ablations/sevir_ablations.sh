#!/bin/bash
# ============================================================
# Ablation Study — SEVIR
# All ablations run sequentially on GPU 0
# Best SEVIR params fixed throughout
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"
SEED=0
GPU=2

# ── Best SEVIR params ─────────────────────────────────────────
DATASET="sevir_lr_latent_32"
SEQ_LEN=25; FRAMES_IN=5; FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"
EXP_DIR="ablations_sevir"
EPOCHS=50

WAVE="db6";     LEVEL=2;    HF_MODE="separate"
BLOCKS=4;       FACTOR=4;   K=3;    SPARSITY=0.01
WS_LOW=0.1;     WS_HIGH=1.0
A_LOW=1.0;      A_HIGH=1.0
B_LOW=0.17;     B_HIGH=0.17
F_LOW=0.1;      F_HIGH=4.0

# ── Ablation backbone names ───────────────────────────────────
ABL1="amplinet_latent_falfcl_only_wavelet_final"
ABL2A="amplinet_latent_falfcl_only_Gabor_ablation_1_final"
ABL2B="amplinet_latent_falfcl_only_Gabor_ablation_2_final"
ABL3A="amplinet_latent_falfcl_only_conv_spectral_afno_final"
ABL3B="amplinet_latent_falfcl_only_conv_spectral_grouped_cnn_final"
ABL3C="amplinet_latent_falfcl_only_conv_spectral_cm_final"
ABL4="amplinet_latent_falfcl_only_spatiotemporal_final"
ABL5="amplinet_latent_falfcl_only_wavelet_spatiotemporal_final"
ABL6="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnoconstgabor_final"
ABL7="amplinet_latent_falfcl_only_IAMAF_final"
ABL8="amplinet_latent_falfcl_only_lf_final"
ABL9="amplinet_latent_falfcl_only_hf_final"
ABL10="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_mse_final"
# ─────────────────────────────────────────────────────────────
run_experiment() {
    local BACKBONE=$1
    local NOTE=$2

    local TAG="${NOTE}_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "  GPU ${GPU} | SEVIR | ${NOTE}"
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
        --res_opt \
        --wavelet_level ${LEVEL} \
        --hf_mode ${HF_MODE} \
        --afno_blocks ${BLOCKS} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --wandb_state 'offline' \
        --wandb_project_name 'Nowcasting_ablations' \
        --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

    # ── Eval ──
    # CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
    #     --backbone ${BACKBONE} \
    #     --dataset ${DATASET} \
    #     --exp_dir ${EXP_DIR} \
    #     --exp_note "${TAG}" \
    #     --ae_ckpt_path "${AE_CKPT}" \
    #     --eval \
    #     --seed ${SEED} \
    #     --seq_len ${SEQ_LEN} \
    #     --frames_in ${FRAMES_IN} \
    #     --frames_out ${FRAMES_OUT} \
    #     --weight_scale_low ${WS_LOW} \
    #     --alpha_low ${A_LOW} \
    #     --beta_low ${B_LOW} \
    #     --freq_multiplier_low ${F_LOW} \
    #     --weight_scale_high ${WS_HIGH} \
    #     --alpha_high ${A_HIGH} \
    #     --beta_high ${B_HIGH} \
    #     --freq_multiplier_high ${F_HIGH} \
    #     --wave ${WAVE} \
    #     --wavelet_level ${LEVEL} \
    #     --hf_mode ${HF_MODE} \
    #     --afno_blocks ${BLOCKS} \
    #     --afno2D_hidden_size_factor ${FACTOR} \
    #     --afno_sparsity_threshold ${SPARSITY} \
    #     --conv_kernel ${K} \
    #     --num_workers 8 \
    #     --wandb_state 'offline'

    echo "  Done: SEVIR | ${NOTE}"
    echo ""
}

echo "=============================================="
echo "  Ablation Study — SEVIR (GPU 0, sequential)"
echo "=============================================="
echo ""

# run_experiment ${ABL1}  "abl1_no_wavelet"
# run_experiment ${ABL2A} "abl2a_no_gabor_filter"
# run_experiment ${ABL2B} "abl2b_gabor_replaced_mlp"
# run_experiment ${ABL3A} "abl3a_no_afno"
# run_experiment ${ABL3B} "abl3b_no_dwconv"
# run_experiment ${ABL3C} "abl3c_no_pwconv"
# run_experiment ${ABL4}  "abl4_no_conv_spectral"
# run_experiment ${ABL5}  "abl5_no_wavelet_conv_spectral"
# run_experiment ${ABL6}  "Original_model_with_const_gabor_params"
# run_experiment ${ABL7}  "abl7_no_IAMAF_module"
# run_experiment ${ABL9}  "abl9_no_hf_wavelet_comp"
run_experiment ${ABL10}  "amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_mse_final"
echo "=============================================="
echo "  SEVIR ablations complete. Check wandb."
echo "=============================================="
