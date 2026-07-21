#!/bin/bash
# ============================================================
# CIKM — DAWNCast, DAWNCast2, DAWNCast3
# GPU 0 → DAWNCast
# GPU 1 → DAWNCast2
# GPU 2 → DAWNCast3
# All 3 in parallel
# Best CIKM params fixed
# ============================================================

# ── Best CIKM params ──────────────────────────────────────────
DATASET="cikm_latent_32"
SEQ_LEN=15; FRAMES_IN=5; FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EPOCHS=50; HF_MODE="separate"
WAVE="db4";   LEVEL=2
BLOCKS=1;     FACTOR=1;  SPARSITY=0.01;  K=7
WS_LOW=0.1;   WS_HIGH=0.25
A_LOW=1.0;    A_HIGH=1.0
B_LOW=43.1034;    B_HIGH=4.8193
F_LOW=22.74;    F_HIGH=24.34

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local BACKBONE=$2
    local EXP_DIR=$3

    echo "=============================================="
    echo "  GPU ${GPU} | CIKM | ${BACKBONE}"
    echo "=============================================="


    # ── Eval ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/Converted_DAWNCAST/cikm/amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final_cikm_latent_32_freq_22.74_24.34_cikm_betas_43.1034_4.8193/ckpt-best.pt\
        --epochs ${EPOCHS} \
        --ae_ckpt_path "${AE_CKPT}" \
        --eval \
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
        --afno_blocks ${BLOCKS} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --afno_sparsity_threshold ${SPARSITY} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --hf_mode ${HF_MODE} \
        --wandb_state 'offline'

    echo "  Done: CIKM | ${BACKBONE}"; echo ""
}

echo "=============================================="
echo "  CIKM — DAWNCast variants (3 GPUs parallel)"
echo "  GPU 0 → DAWNCast"
echo "  GPU 1 → DAWNCast2"
echo "  GPU 2 → DAWNCast3"
echo "=============================================="
echo ""

run_experiment 1 DAWNCast  DAWNCast_newenv  &
PID_GPU0=$!



wait ${PID_GPU0}
echo "GPU 0 complete! (DAWNCast)"


echo ""
echo "=============================================="
echo "  CIKM DAWNCast runs complete. Check wandb."
echo "=============================================="