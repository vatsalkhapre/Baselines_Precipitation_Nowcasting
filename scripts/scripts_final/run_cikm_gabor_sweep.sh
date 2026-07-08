

#!/bin/bash
# ============================================================
# CIKM 20 Off-Diagonal Frequency Sweep
# Runs the 20 non-diagonal (flow, fhigh) combinations.
# GPU0 and GPU1 each process 10 jobs sequentially in parallel.
# ============================================================

BACKBONE="amplinet_latent_falfcl_only_2_3_13_2_AFNO2D_relu_convparallelwaveletafnogabor_final"
RUNNER="run_alphapre_convlstm_sevir_lr_latent_model_novelty.py"

DATASET="cikm_latent_32"
SEQ_LEN=15
FRAMES_IN=5
FRAMES_OUT=10
AE_CKPT="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth"
EXP_DIR="multiseed_cikm"
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

    local TAG="freq_${F_LOW}_${F_HIGH}_cikm_betas_${B_LOW}_${B_HIGH}"
    local DS_SHORT=$(echo ${DATASET} | cut -d'_' -f1)

    echo "=============================================="
    echo "GPU ${GPU} | CIKM | flow=${F_LOW} | fhigh=${F_HIGH}"
    echo "=============================================="

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
        --wandb_project_name DAWNCAST_Gabor_sweep \
        --run_name "CIKM_${BACKBONE}_${DS_SHORT}_${TAG}"

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

FLOWS=(22.74)
FHIGHS=(9.13 24.34)

# GPU0_TASKS=()
# GPU1_TASKS=()

count=0
for i in "${!FLOWS[@]}"; do
  for j in "${!FHIGHS[@]}"; do
    # [[ $i -eq $j ]] && continue
    if (( count % 2 == 0 )); then
      GPU0_TASKS+=("${FLOWS[$i]} ${FHIGHS[$j]}")
    else
      GPU1_TASKS+=("${FLOWS[$i]} ${FHIGHS[$j]}")
    fi
    ((count++))
  done
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

echo "Starting CIKM 20 off-diagonal sweep..."

run_gpu0 &
PID0=$!
run_gpu1 &
PID1=$!

wait $PID0
echo "GPU 0 complete."

wait $PID1
echo "GPU 1 complete."

echo "All CIKM off-diagonal runs finished."
