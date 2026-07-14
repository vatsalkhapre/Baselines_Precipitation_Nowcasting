

#!/bin/bash
# ============================================================
# CIKM 20 Off-Diagonal Frequency Sweep
# Runs the 20 non-diagonal (flow, fhigh) combinations.
# GPU0 and GPU1 each process 10 jobs sequentially in parallel.
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_expgabor_final"
RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novel_ablations.py"

DATASET="cikm_latent_32"
SEQ_LEN=15
FRAMES_IN=5
FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="gabor_exp_cikm"
EPOCHS=50
SEED=0

WAVE="db4"
LEVEL=2
HF_MODE="separate"

BLOCKS=1
FACTOR=1
K=7
SPARSITY=0.01

WS_LOW=0.1
WS_HIGH=0.25
A_LOW=1.0
A_HIGH=1.0
B_LOW=43.1034
B_HIGH=4.8193

run_experiment() {
    local GPU=$1
    local F_LOW=$2
    local F_HIGH=$3

    local TAG="CIKM_flow${F_LOW}_fhigh${F_HIGH}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "GPU ${GPU} | CIKM | flow=${F_LOW} | fhigh=${F_HIGH}"
    echo "=============================================="

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
    #     --wandb_state online \
    #     --wandb_project_name DAWNCAST_Gabor_sweep \
    #     --run_name "CIKM_${BACKBONE}_${DS_SHORT}_${TAG}"

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

# FLOWS=(181.94 363.89 714.49)
# FHIGHS=(3.04 9.13 24.34 48.67 95.56)

FLOWS=(181.94)
FHIGHS=(9.13)

# -------------------------------------------------
# Build task list while skipping requested pairs
# -------------------------------------------------
TASKS=()

for F_LOW in "${FLOWS[@]}"; do
    for F_HIGH in "${FHIGHS[@]}"; do

        # Skip these two combinations
        if [[ "$F_LOW" == "181.94" && "$F_HIGH" == "3.04" ]]; then
            continue
        fi

        TASKS+=("$F_LOW $F_HIGH")
    done
done

GPU0_TASKS=()
GPU1_TASKS=()

# Alternate tasks between GPUs
for ((i=0; i<${#TASKS[@]}; i++)); do
    if (( i % 2 == 0 )); then
        GPU0_TASKS+=("${TASKS[$i]}")
    else
        GPU1_TASKS+=("${TASKS[$i]}")
    fi
done

run_gpu0() {
    for task in "${GPU0_TASKS[@]}"; do
        read F_LOW F_HIGH <<< "$task"
        run_experiment 0 "$F_LOW" "$F_HIGH"
    done
}

run_gpu1() {
    for task in "${GPU1_TASKS[@]}"; do
        read F_LOW F_HIGH <<< "$task"
        run_experiment 1 "$F_LOW" "$F_HIGH"
    done
}

echo "=============================================="
echo "GPU0 will run ${#GPU0_TASKS[@]} jobs"
echo "GPU1 will run ${#GPU1_TASKS[@]} jobs"
echo "=============================================="

run_gpu0 &
PID0=$!

run_gpu1 &
PID1=$!

wait $PID0
echo "GPU0 finished."

wait $PID1
echo "GPU1 finished."

echo "All experiments completed."