#!/bin/bash
# ============================================================
# SEVIR — Config 1 (near-MLP regime)
# GPU 1 only
# W_low:0.1, W_high:0.25, f_low:0.1, f_high:0.1
# beta:100, alpha:1.0, db4, Level:2, k:7, blocks:1, factor:1, sparsity:0.01
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novelty.py"
BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
SEED=0
GPU=1

# ── Dataset ───────────────────────────────────────────────────
DATASET="sevir_lr_latent_32"
SEQ_LEN=25; FRAMES_IN=5; FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"
EXP_DIR="sevir_final_config1"
EPOCHS=50

# ── Config 1 params ───────────────────────────────────────────
WAVE="db4";   LEVEL=2;   HF_MODE="separate"
BLOCKS=1;     FACTOR=1;  K=7;   SPARSITY=0.01
WS_LOW=0.1;   WS_HIGH=0.25
A_LOW=1.0;    A_HIGH=1.0
B_LOW=100;    B_HIGH=100
F_LOW=0.1;    F_HIGH=0.1

TAG="config1_flow${F_LOW}_fhigh${F_HIGH}_b${B_LOW}_${WAVE}_J${LEVEL}_${HF_MODE}"
DS_SHORT="sevir"

echo "=============================================="
echo "  SEVIR | Config 1 | GPU ${GPU}"
echo "  wave=${WAVE} J=${LEVEL} blocks=${BLOCKS} factor=${FACTOR}"
echo "  f_low=${F_LOW} f_high=${F_HIGH} beta=${B_LOW}"
echo "=============================================="
echo ""

# ── Train ──
# CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
#     --backbone ${BACKBONE} \
#     --dataset ${DATASET} \
#     --exp_dir ${EXP_DIR} \
#     --exp_note "${TAG}" \
#     --epochs ${EPOCHS} \
#     --ae_ckpt_path "${AE_CKPT}" \
#     --valid \
#     --seq_len ${SEQ_LEN} \
#     --seed ${SEED} \
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
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name "${BACKBONE}_${DS_SHORT}_${TAG}"

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

echo ""
echo "=============================================="
echo "  SEVIR Config 1 complete. Check wandb."
echo "=============================================="
