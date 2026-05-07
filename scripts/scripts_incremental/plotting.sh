#!/bin/bash
# ============================================================
# Plotting — Incremental Ablation (5 models x 4 datasets)
# GPU 0 → CIKM + Shanghai
# GPU 1 → MeteoNet + SEVIR
# All sequential within each GPU, both GPUs in parallel
#
# UPDATE: ckpt paths below before running
# ============================================================

RUNNER="run_alphapre_convlstm_sevir_lr_latent_with_plotting_consecutive_only5.py"
PLOT_STRIDE=20

# ── Backbone names ────────────────────────────────────────────
ABL1="amplinet_latent_falfcl_only_incr1_mlp_only_final"
ABL2="amplinet_latent_falfcl_only_incr2_mlp_gabor_final"
ABL3="amplinet_latent_falfcl_only_incr3_mlp_gabor_wavelet_final"
ABL4="amplinet_latent_falfcl_only_incr3p5_mlp_gabor_wavelet_afno_only_final"
ABL5="amplinet_latent_falfcl_only_incr4_mlp_gabor_wavelet_conv_final"

# ── AE checkpoints ────────────────────────────────────────────
AE_CIKM="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
AE_SHANGHAI="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth"
AE_METEONET="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth"
AE_SEVIR="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"

# ── Model checkpoints — UPDATE THESE ─────────────────────────
# Format: CKPT_<DATASET>_<INC>

# CIKM
CKPT_CIKM_1="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/CIKM/amplinet_latent_falfcl_only_incr1_mlp_only_final_cikm_latent_32_inc1_db4_J2_separate/checkpoints/ckpt-best.pt"
CKPT_CIKM_2="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/CIKM/amplinet_latent_falfcl_only_incr2_mlp_gabor_final_cikm_latent_32_inc2_db4_J2_separate/checkpoints/ckpt-best.pt"
CKPT_CIKM_3="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/CIKM/amplinet_latent_falfcl_only_incr3_mlp_gabor_wavelet_final_cikm_latent_32_inc3_db4_J2_separate/checkpoints/ckpt-best.pt"
CKPT_CIKM_4="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/CIKM/amplinet_latent_falfcl_only_incr3p5_mlp_gabor_wavelet_afno_only_final_cikm_latent_32_inc4_db4_J2_separate/checkpoints/ckpt-best.pt"
CKPT_CIKM_5="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/CIKM/amplinet_latent_falfcl_only_incr4_mlp_gabor_wavelet_conv_final_cikm_latent_32_inc5_db4_J2_separate/checkpoints/ckpt-best.pt"

# Shanghai
CKPT_SHANGHAI_1="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Shanghai/amplinet_latent_falfcl_only_incr1_mlp_only_final_shanghai_lr_latent_32_inc1_db6_J3_separate_0.17_4.0/checkpoints/ckpt-best.pt"
CKPT_SHANGHAI_2="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Shanghai/amplinet_latent_falfcl_only_incr2_mlp_gabor_final_shanghai_lr_latent_32_inc2_db6_J3_separate_0.17_4.0/checkpoints/ckpt-best.pt"
CKPT_SHANGHAI_3="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Shanghai/amplinet_latent_falfcl_only_incr3_mlp_gabor_wavelet_final_shanghai_lr_latent_32_inc3_db6_J3_separate_0.17_4.0/checkpoints/ckpt-best.pt"
CKPT_SHANGHAI_4="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Shanghai/amplinet_latent_falfcl_only_incr3p5_mlp_gabor_wavelet_afno_only_final_shanghai_lr_latent_32_inc4_db6_J3_separate_0.17_4.0/checkpoints/ckpt-best.pt"
CKPT_SHANGHAI_5="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Shanghai/amplinet_latent_falfcl_only_incr4_mlp_gabor_wavelet_conv_final_shanghai_lr_latent_32_inc5_db6_J3_separate_0.17_4.0/checkpoints/ckpt-best.pt"
                                                                        

# MeteoNet
CKPT_METEONET_1="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Meteonet/amplinet_latent_falfcl_only_incr1_mlp_only_final_meteo_lr_latent_32_inc1_db6_J1_separate/checkpoints/ckpt-best.pt"
CKPT_METEONET_2="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Meteonet/amplinet_latent_falfcl_only_incr2_mlp_gabor_final_meteo_lr_latent_32_inc2_db6_J1_separate/checkpoints/ckpt-best.pt"
CKPT_METEONET_3="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Meteonet/amplinet_latent_falfcl_only_incr3_mlp_gabor_wavelet_final_meteo_lr_latent_32_inc3_db6_J1_separate/checkpoints/ckpt-best.pt"
CKPT_METEONET_4="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Meteonet/amplinet_latent_falfcl_only_incr3p5_mlp_gabor_wavelet_afno_only_final_meteo_lr_latent_32_inc4_db6_J1_separate/checkpoints/ckpt-best.pt"
CKPT_METEONET_5="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Meteonet/amplinet_latent_falfcl_only_incr4_mlp_gabor_wavelet_conv_final_meteo_lr_latent_32_inc5_db6_J1_separate/checkpoints/ckpt-best.pt"

# SEVIR
CKPT_SEVIR_1="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Sevir/amplinet_latent_falfcl_only_incr1_mlp_only_final_sevir_lr_latent_32_inc_1_db6_J2_separate/checkpoints/ckpt-best.pt"
CKPT_SEVIR_2="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Sevir/amplinet_latent_falfcl_only_incr2_mlp_gabor_final_sevir_lr_latent_32_inc_2_db6_J2_separate/checkpoints/ckpt-best.pt"
CKPT_SEVIR_3="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Sevir/amplinet_latent_falfcl_only_incr3_mlp_gabor_wavelet_final_sevir_lr_latent_32_inc_3_db6_J2_separate/checkpoints/ckpt-best.pt"
CKPT_SEVIR_4="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Sevir/amplinet_latent_falfcl_only_incr3p5_mlp_gabor_wavelet_afno_only_final_sevir_lr_latent_32_inc_4_db6_J2_separate/checkpoints/ckpt-best.pt"
CKPT_SEVIR_5="/home/vatsal/Dataserver2/Neurips/Ablations/Incremental/Sevir/amplinet_latent_falfcl_only_incr4_mlp_gabor_wavelet_conv_final_sevir_lr_latent_32_inc_5_db6_J2_separate/checkpoints/ckpt-best.pt"

# ─────────────────────────────────────────────────────────────
run_plot() {
    local GPU=$1
    local BACKBONE=$2
    local CKPT=$3
    local AE_CKPT=$4
    local DATASET=$5
    local SEQ_LEN=$6
    local FRAMES_IN=$7
    local FRAMES_OUT=$8
    local WAVE=$9
    local LEVEL=${10}
    local BLOCKS=${11}
    local FACTOR=${12}
    local SPARSITY=${13}
    local K=${14}
    local WS_LOW=${15}; local WS_HIGH=${16}
    local A_LOW=${17};  local A_HIGH=${18}
    local B_LOW=${19};  local B_HIGH=${20}
    local F_LOW=${21};  local F_HIGH=${22}
    local NOTE=${23}

    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)
    echo "=============================================="
    echo "  GPU ${GPU} | ${DS_SHORT} | ${NOTE}"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES=${GPU} python3 ${RUNNER} \
        --backbone ${BACKBONE} \
        --dataset ${DATASET} \
        --eval \
        --plot \
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
        --hidden_dim 64 \
        --afno_blocks ${BLOCKS} \
        --afno_sparsity_threshold ${SPARSITY} \
        --afno2D_hidden_size_factor ${FACTOR} \
        --conv_kernel ${K} \
        --num_workers 8 \
        --plot_stride ${PLOT_STRIDE} \
        --ckpt_milestone "${CKPT}" \
        --ae_ckpt_path "${AE_CKPT}" \
        --wandb_state 'offline'

    echo "  Done: ${DS_SHORT} | ${NOTE}"; echo ""
}

# ─────────────────────────────────────────────────────────────
# GPU 0 → CIKM (5 models) then Shanghai (5 models)
# GPU 1 → MeteoNet (5 models) then SEVIR (5 models)
# ─────────────────────────────────────────────────────────────

# run_gpu0() {
#     echo "=== GPU 0: CIKM ==="
#     run_plot 0 ${ABL1} ${CKPT_CIKM_1} ${AE_CIKM} cikm_latent_32    15 5 10 db4 2 1 1 0.01 7 0.1 0.25 1.0 1.0 100  100  0.1 0.1 "inc1_cikm"
#     run_plot 0 ${ABL2} ${CKPT_CIKM_2} ${AE_CIKM} cikm_latent_32    15 5 10 db4 2 1 1 0.01 7 0.1 0.25 1.0 1.0 100  100  0.1 0.1 "inc2_cikm"
#     run_plot 0 ${ABL3} ${CKPT_CIKM_3} ${AE_CIKM} cikm_latent_32    15 5 10 db4 2 1 1 0.01 7 0.1 0.25 1.0 1.0 100  100  0.1 0.1 "inc3_cikm"
#     run_plot 0 ${ABL4} ${CKPT_CIKM_4} ${AE_CIKM} cikm_latent_32    15 5 10 db4 2 1 1 0.01 7 0.1 0.25 1.0 1.0 100  100  0.1 0.1 "inc4_cikm"
#     run_plot 0 ${ABL5} ${CKPT_CIKM_5} ${AE_CIKM} cikm_latent_32    15 5 10 db4 2 1 1 0.01 7 0.1 0.25 1.0 1.0 100  100  0.1 0.1 "inc5_cikm"

#     echo "=== GPU 0: Shanghai ==="
#     run_plot 0 ${ABL1} ${CKPT_SHANGHAI_1} ${AE_SHANGHAI} shanghai_lr_latent_32 25 5 20 db6 3 4 3 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 4.0 4.0 "inc1_shanghai"
#     run_plot 0 ${ABL2} ${CKPT_SHANGHAI_2} ${AE_SHANGHAI} shanghai_lr_latent_32 25 5 20 db6 3 4 3 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 4.0 4.0 "inc2_shanghai"
#     run_plot 0 ${ABL3} ${CKPT_SHANGHAI_3} ${AE_SHANGHAI} shanghai_lr_latent_32 25 5 20 db6 3 4 3 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 4.0 4.0 "inc3_shanghai"
#     run_plot 0 ${ABL4} ${CKPT_SHANGHAI_4} ${AE_SHANGHAI} shanghai_lr_latent_32 25 5 20 db6 3 4 3 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 4.0 4.0 "inc4_shanghai"
#     run_plot 0 ${ABL5} ${CKPT_SHANGHAI_5} ${AE_SHANGHAI} shanghai_lr_latent_32 25 5 20 db6 3 4 3 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 4.0 4.0 "inc5_shanghai"
# }

run_gpu1() {
    echo "=== GPU 1: MeteoNet ==="
    run_plot 1 ${ABL1} ${CKPT_METEONET_1} ${AE_METEONET} meteo_lr_latent_32   25 5 20 db6 1 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc1_meteonet"
    run_plot 1 ${ABL2} ${CKPT_METEONET_2} ${AE_METEONET} meteo_lr_latent_32   25 5 20 db6 1 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc2_meteonet"
    run_plot 1 ${ABL3} ${CKPT_METEONET_3} ${AE_METEONET} meteo_lr_latent_32   25 5 20 db6 1 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc3_meteonet"
    run_plot 1 ${ABL4} ${CKPT_METEONET_4} ${AE_METEONET} meteo_lr_latent_32   25 5 20 db6 1 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc4_meteonet"
    run_plot 1 ${ABL5} ${CKPT_METEONET_5} ${AE_METEONET} meteo_lr_latent_32   25 5 20 db6 1 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc5_meteonet"

    # echo "=== GPU 1: SEVIR ==="
    # run_plot 1 ${ABL1} ${CKPT_SEVIR_1} ${AE_SEVIR} sevir_lr_latent_32    25 5 20 db6 2 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc1_sevir"
    # run_plot 1 ${ABL2} ${CKPT_SEVIR_2} ${AE_SEVIR} sevir_lr_latent_32    25 5 20 db6 2 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc2_sevir"
    # run_plot 1 ${ABL3} ${CKPT_SEVIR_3} ${AE_SEVIR} sevir_lr_latent_32    25 5 20 db6 2 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc3_sevir"
    # run_plot 1 ${ABL4} ${CKPT_SEVIR_4} ${AE_SEVIR} sevir_lr_latent_32    25 5 20 db6 2 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc4_sevir"
    # run_plot 1 ${ABL5} ${CKPT_SEVIR_5} ${AE_SEVIR} sevir_lr_latent_32    25 5 20 db6 2 4 4 0.01 3 0.1 1.0 1.0 1.0 0.17 0.17 0.1 4.0 "inc5_sevir"
}

echo "=============================================="
echo "  Incremental Plotting — 5 models x 4 datasets"
echo "  GPU 0 → CIKM + Shanghai (10 runs)"
echo "  GPU 1 → MeteoNet + SEVIR (10 runs)"
echo "=============================================="
echo ""

run_gpu0 &
PID_GPU0=$!

run_gpu1 &
PID_GPU1=$!

# wait ${PID_GPU0}
# echo "GPU 0 complete! (CIKM + Shanghai)"

wait ${PID_GPU1}
echo "GPU 1 complete! (MeteoNet + SEVIR)"

echo ""
echo "=============================================="
echo "  All plotting complete."
echo "=============================================="