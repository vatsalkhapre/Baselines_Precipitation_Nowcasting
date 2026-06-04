

#!/bin/bash
# ============================================================
<<<<<<<< HEAD:scripts/scripts_final/run_cikm_statstical_analysis.sh
# Multi-Seed Runs — CIKM
# Seeds: 1, 2, 3, 4 (seed=0 already done)
# GPU 0 → seed 1, seed 2 (sequential)
# GPU 1 → seed 3, seed 4 (sequential)
# Both GPUs in parallel
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novelty.py"

# ── Best CIKM params ──────────────────────────────────────────
DATASET="cikm_latent_32"
SEQ_LEN=15; FRAMES_IN=5; FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="multiseed_cikm"
EPOCHS=50

WAVE="db4";   LEVEL=2;   HF_MODE="separate"
BLOCKS=1;     FACTOR=1;  K=7;   SPARSITY=0.01
WS_LOW=0.1;   WS_HIGH=0.25
A_LOW=1.0;    A_HIGH=1.0
B_LOW=100;    B_HIGH=100
F_LOW=0.1;    F_HIGH=0.1
========
# Ablation Study — MeteoNet
# 7 ablations distributed across 3 GPUs
# GPU 0 → Abl 1, 2a, 2b  (sequential)
# GPU 1 → Abl 3a, 3b     (sequential)
# GPU 2 → Abl 3c, 4      (sequential)
# All GPUs run in parallel
# Best MeteoNet params fixed throughout
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"
SEED=0

# ── Best MeteoNet params ──────────────────────────────────────
DATASET="meteo_lr_latent_32"
SEQ_LEN=25; FRAMES_IN=5; FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
EXP_DIR="ablations_meteonet"
EPOCHS=50

WAVE="db6";     LEVEL=1;    HF_MODE="separate"
BLOCKS=4;       FACTOR=4;   K=3;    SPARSITY=0.01
WS_LOW=0.1;     WS_HIGH=1.0
A_LOW=1.0;      A_HIGH=1.0
B_LOW=0.17;     B_HIGH=0.17
F_LOW=0.1;      F_HIGH=4.0
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_ablations/meteo_ablations.sh

# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
<<<<<<<< HEAD:scripts/scripts_final/run_cikm_statstical_analysis.sh
    local SEED=$2
========
    local BACKBONE=$2
    local NOTE=$3
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_ablations/meteo_ablations.sh

    local TAG="seed${SEED}_${WAVE}_J${LEVEL}_${HF_MODE}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
<<<<<<<< HEAD:scripts/scripts_final/run_cikm_statstical_analysis.sh
    echo "  GPU ${GPU} | CIKM | seed=${SEED}"
========
    echo "  GPU ${GPU} | MeteoNet | ${NOTE}"
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_ablations/meteo_ablations.sh
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
        --wandb_state 'offline' \
        --wandb_project_name 'Alphapre' \
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

<<<<<<<< HEAD:scripts/scripts_final/run_cikm_statstical_analysis.sh
    echo "  Done: CIKM | seed=${SEED}"; echo ""
}

# ─────────────────────────────────────────────────────────────
run_gpu0() {
    run_experiment 0 1
    run_experiment 0 2
}

run_gpu1() {
    run_experiment 1 3
    run_experiment 1 4
========
    echo "  Done: MeteoNet | ${NOTE}"
    echo ""
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_ablations/meteo_ablations.sh
}

# ─────────────────────────────────────────────────────────────
# GPU 0 → Abl 1, 2a, 2b
# GPU 1 → Abl 3a, 3b
# GPU 2 → Abl 3c, 4
# ─────────────────────────────────────────────────────────────

run_gpu0() {
    # run_experiment 0 ${ABL1}  "abl1_no_wavelet"
    # run_experiment 0 ${ABL2A} "abl2a_no_gabor_filter"
    # run_experiment 0 ${ABL2B} "abl2b_gabor_replaced_mlp"
    # run_experiment 0 ${ABL5}  "abl5_no_wavelet_conv_spectral"
    run_experiment 0 ${ABL6}  "Original_model_with_const_gabor_params"

}

# run_gpu1() {
#     run_experiment 1 ${ABL3A} "abl3a_no_afno"
#     run_experiment 1 ${ABL3B} "abl3b_no_dwconv"
# }

# run_gpu2() {
#     run_experiment 2 ${ABL3C} "abl3c_no_pwconv"
#     run_experiment 2 ${ABL4}  "abl4_no_conv_spectral"
# }

echo "=============================================="
<<<<<<<< HEAD:scripts/scripts_final/run_cikm_statstical_analysis.sh
echo "  Multi-Seed — CIKM (2 GPUs parallel)"
echo "  GPU 0 → seed 1, seed 2"
echo "  GPU 1 → seed 3, seed 4"
========
echo "  Ablation Study — MeteoNet (3 GPUs parallel)"
echo "  GPU 0 → Abl 1, 2a, 2b"
echo "  GPU 1 → Abl 3a, 3b"
echo "  GPU 2 → Abl 3c, 4"
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_ablations/meteo_ablations.sh
echo "=============================================="
echo ""

run_gpu0 &
PID_GPU0=$!

<<<<<<<< HEAD:scripts/scripts_final/run_cikm_statstical_analysis.sh
run_gpu1 &
PID_GPU1=$!

wait ${PID_GPU0}
echo "GPU 0 complete! (seed 1, 2)"

wait ${PID_GPU1}
echo "GPU 1 complete! (seed 3, 4)"

echo ""
echo "=============================================="
echo "  CIKM multi-seed runs complete. Check wandb."
========
# run_gpu1 &
# PID_GPU1=$!

# run_gpu2 &
# PID_GPU2=$!

wait ${PID_GPU0}
echo "GPU 0 complete! (Abl 1, 2a, 2b)"

# wait ${PID_GPU1}
# echo "GPU 1 complete! (Abl 3a, 3b)"

# wait ${PID_GPU2}
# echo "GPU 2 complete! (Abl 3c, 4)"

echo ""
echo "=============================================="
echo "  MeteoNet ablations complete. Check wandb."
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_ablations/meteo_ablations.sh
echo "=============================================="