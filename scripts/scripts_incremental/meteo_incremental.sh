#!/bin/bash
# ============================================================
<<<<<<<< HEAD:scripts/scripts_incremental/shanghai.sh
# Ablation Study — SHANGHAI
# All 7 ablations run sequentially on GPU 0
# Best SHANGHAI params fixed throughout
========
# Ablation Study — Meteonet
# All 7 ablations run sequentially on GPU 0
# Best Meteonet params fixed throughout
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_incremental/meteo_incremental.sh
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"
SEED=0
GPU=0

<<<<<<< HEAD
<<<<<<<< HEAD:scripts/scripts_incremental/shanghai.sh
# ── Best SHANGHAI params ──────────────────────────────────────────
DATASET="shanghai_lr_latent_32"
SEQ_LEN=25; FRAMES_IN=5; FRAMES_OUT=20
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth"
EXP_DIR="incremental_shanghai"
EPOCHS=50

WAVE="db6";     LEVEL=3;    HF_MODE="separate"
BLOCKS=4;       FACTOR=3;   K=3;    SPARSITY=0.01
WS_LOW=0.1;     WS_HIGH=1.0
A_LOW=1.0;      A_HIGH=1.0
B_LOW=0.17;     B_HIGH=0.17
F_LOW=4.0;      F_HIGH=4.0
========
<<<<<<< HEAD:scripts/scripts_incremental/cikm.sh
# ── Best CIKM params ──────────────────────────────────────────
DATASET="cikm_latent_32"
SEQ_LEN=15; FRAMES_IN=5; FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="Incremental_cikm"
=======
=======
>>>>>>> e5e9701 (ablations)
# ── Best Meteonet params ──────────────────────────────────────────
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
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_incremental/meteo_incremental.sh

# ── Ablation backbone names ───────────────────────────────────
ABL1="amplinet_latent_falfcl_only_incr1_mlp_only_final"
ABL2="amplinet_latent_falfcl_only_incr2_mlp_gabor_final"
ABL3="amplinet_latent_falfcl_only_incr3_mlp_gabor_wavelet_final"
ABL4="amplinet_latent_falfcl_only_incr3p5_mlp_gabor_wavelet_afno_only_final"
ABL5="amplinet_latent_falfcl_only_incr4_mlp_gabor_wavelet_conv_final"
# ─────────────────────────────────────────────────────────────
run_experiment() {
    local GPU=$1
    local BACKBONE=$2
    local NOTE=$3

    local TAG="${NOTE}_${WAVE}_J${LEVEL}_${HF_MODE}_${B_LOW}_${F_LOW}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
<<<<<<<< HEAD:scripts/scripts_incremental/shanghai.sh
    echo "  GPU ${GPU} | SHANGHAI | ${NOTE}"
========
    echo "  GPU ${GPU} | Meteonet | ${NOTE}"
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_incremental/meteo_incremental.sh
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

<<<<<<<< HEAD:scripts/scripts_incremental/shanghai.sh
    echo "  Done: SHANGHAI | ${NOTE}"
========
    echo "  Done: Meteonet | ${NOTE}"
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_incremental/meteo_incremental.sh
    echo ""
}
run_gpu0() {
    run_experiment 0 ${ABL1}  "inc 1"
    run_experiment 0 ${ABL2} "inc 2"
}
run_gpu1() {
    run_experiment 1 ${ABL3} "inc 3"
    run_experiment 1 ${ABL4}  "inc 4"
}
run_gpu2() {
    run_experiment 2 ${ABL5}  "inc 5"
}
echo "=============================================="
<<<<<<<< HEAD:scripts/scripts_incremental/shanghai.sh
echo "  Ablation Study — SHANGHAI (GPU 0, sequential)"
========
echo "  Ablation Study — Meteonet (GPU 0, sequential)"
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_incremental/meteo_incremental.sh
echo "  7 ablations total"
echo "=============================================="
echo ""

<<<<<<<< HEAD:scripts/scripts_incremental/shanghai.sh

run_gpu0() {
    run_experiment 0 ${ABL1}  "inc 1"
    run_experiment 0 ${ABL2} "inc 2"
    run_experiment 0 ${ABL3} "inc 3"
}

run_gpu1() {
    run_experiment 1 ${ABL4}  "inc 4"
    run_experiment 1 ${ABL5}  "inc 5"
}

========
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_incremental/meteo_incremental.sh
run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

<<<<<<<< HEAD:scripts/scripts_incremental/shanghai.sh
echo "=============================================="
echo "  SHANGHAI ablations complete. Check wandb."
========
run_gpu2 &
PID_GPU2=$!

wait ${PID_GPU0}
echo "GPU 0 complete!"
wait ${PID_GPU1}
echo "GPU 1 complete!"

wait ${PID_GPU2}
echo "GPU 2 complete!"
echo "=============================================="
echo "  Meteonet ablations complete. Check wandb."
>>>>>>>> 655fe94 (Incremental ablations):scripts/scripts_incremental/meteo_incremental.sh
echo "=============================================="