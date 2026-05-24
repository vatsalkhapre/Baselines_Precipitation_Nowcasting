#!/bin/bash
# ============================================================
# MeteoNet — DAWNCast, DAWNCast2, DAWNCast3
# GPU 0 → DAWNCast
# GPU 1 → DAWNCast2
# GPU 2 → DAWNCast3
# All 3 in parallel
# Best MeteoNet params fixed
# ============================================================

# ── Best MeteoNet params ──────────────────────────────────────
DATASET="meteo_lr_latent_32"
SEQ_LEN=25; FRAMES_IN=5; FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
EPOCHS=50; HF_MODE="separate"

WAVE="db6";   LEVEL=1
BLOCKS=4;     FACTOR=4;  SPARSITY=0.01;  K=3
WS_LOW=0.1;   WS_HIGH=1.0
A_LOW=1.0;    A_HIGH=1.0
B_LOW=0.17;   B_HIGH=0.17
F_LOW=0.1;    F_HIGH=4.0

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local BACKBONE=$2
    local EXP_DIR=$3

    echo "=============================================="
    echo "  GPU ${GPU} | MeteoNet | ${BACKBONE}"
    echo "=============================================="

    # ── Train ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note ${DATASET} \
        --epochs ${EPOCHS} \
        --ae_ckpt_path "${AE_CKPT}" \
        --valid \
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
        --wandb_state 'online' \
        --wandb_project_name 'Alphapre' \
        --run_name ${BACKBONE}

    # ── Eval ──
    CUDA_VISIBLE_DEVICES=${GPU} python3 run_alphapre_convlstm_sevir_lr_latent.py \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --exp_dir ${EXP_DIR} \
        --exp_note ${DATASET} \
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
        --wandb_state 'offline' \

    echo "  Done: MeteoNet | ${BACKBONE}"; echo ""
}

echo "=============================================="
echo "  MeteoNet — DAWNCast variants (3 GPUs parallel)"
echo "  GPU 0 → DAWNCast"
echo "  GPU 1 → DAWNCast2"
echo "  GPU 2 → DAWNCast3"
echo "=============================================="
echo ""

run_experiment 0 DAWNCast  DAWNCast  &
PID_GPU0=$!

run_experiment 1 DAWNCast2 DAWNCast2 &
PID_GPU1=$!

run_experiment 2 DAWNCast3 DAWNCast3 &
PID_GPU2=$!

wait ${PID_GPU0}
echo "GPU 0 complete! (DAWNCast)"

wait ${PID_GPU1}
echo "GPU 1 complete! (DAWNCast2)"

wait ${PID_GPU2}
echo "GPU 2 complete! (DAWNCast3)"

echo ""
echo "=============================================="
echo "  MeteoNet DAWNCast runs complete. Check wandb."
echo "=============================================="