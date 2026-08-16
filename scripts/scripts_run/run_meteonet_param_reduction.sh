#!/bin/bash
# ============================================================
# Meteonet pixel-space PARAMETER-REDUCTION sweep (DAWN-Cast, dawncast.py)
#
# Goal: cut parameter count vs the 59.47M Gabor-sweep baseline without losing
#       test CSI. Gabor init is frozen at the meteonet sweep-best values.
#
# Baseline (MET-B0, already run, NOT re-run here):
#   Exps/Gabor_sweep_runs/Meteonet_pixel_flow1.09_fhigh1.12
#   nb=4 hsf=4 lvl=1 wave=db6 k=3  ->  59,466,369 params
#   best val CSI 0.45304 @ epoch 20/50 ; test CSI 0.45289 (ckpt-best)
#
# Sweep axis: hidden_size_factor / num_blocks, which is the only real capacity
# knob (it carries the whole STR weight term, 32.77M of the baseline's 59.47M).
# Everything else is held at the baseline.
#
# Env split (as specified):  train/val -> earthformer ;  test -> alphapre_manual
# Epochs = 50, matching the baseline exactly so the cosine LR schedule and the
# ckpt-best selection are apples-to-apples.
#
# Usage:
#   nohup scripts/scripts_run/run_meteonet_param_reduction.sh > meteonet_sweep.out 2>&1 &
#   scripts/scripts_run/run_meteonet_param_reduction.sh --dry_run
# ============================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/run_alphapre_convlstm.py"
LOGGER="${REPO_ROOT}/scripts/param_budget_log.py"

TRAIN_ENV="earthformer"
TEST_ENV="alphapre_manual"

BACKBONE="DAWNCast"
DATASET="meteo"
EXP_DIR="meteonet_param_reduction"

# ---- Held at the baseline ------------------------------------
IMG_SIZE=128 ; IMG_CHANNEL=1 ; SEQ_LEN=25
FRAMES_IN=5  ; FRAMES_OUT=20
HIDDEN_DIM=64 ; HF_MODE="separate" ; SIZE_FACTOR=1.0
WAVE="db6" ; WAVELET_LEVEL=1 ; CONV_KERNEL=3
SPARSITY_THRESHOLD=0.01
EPOCHS=50 ; SEED=0 ; BATCH_SIZE=4 ; LR=1e-4 ; NUM_WORKERS=8

# ---- Gabor init: meteonet sweep best (Meteonet_pixel_flow1.09_fhigh1.12) ----
WS_LOW=0.1    ; WS_HIGH=1.0
A_LOW=1.0     ; A_HIGH=1.0
B_LOW=0.0995  ; B_HIGH=0.1643
F_LOW=1.09    ; F_HIGH=1.12
GABOR_SRC="Exps/Gabor_sweep_runs/Meteonet_pixel_flow1.09_fhigh1.12"

BASELINE_PARAMS=59466369

WANDB_STATE="online"
WANDB_PROJECT="DAWNCast_param_budget"

DRY_RUN=0
[[ "${1:-}" == "--dry_run" ]] && DRY_RUN=1

# ---- Sweep: "CONFIG_ID NB HSF EXPECTED_PARAMS GPU" -----------
# Ordered so the two runs that bracket the interesting region start first.
CONFIGS=(
  "MET-R2 16 4 34890369 0"    # ratio 1/4   58.7% of baseline
  "MET-R4 16 1 28707969 1"    # ratio 1/16  48.3%
  "MET-R3 16 2 30768769 1"    # ratio 1/8   51.7%
  "MET-R1  8 4 43082369 0"    # ratio 1/2   72.4%
  "MET-R5 64 1 27171969 0"    # ratio 1/64  45.7%
)

echo "============================================================"
echo " Meteonet parameter-reduction sweep"
echo "============================================================"
echo "  baseline MET-B0 : ${BASELINE_PARAMS} params, test CSI 0.45289"
echo "  train env       : ${TRAIN_ENV}      test env: ${TEST_ENV}"
echo "  epochs          : ${EPOCHS} (matches baseline LR schedule)"
echo "  gabor init      : ws=(${WS_LOW},${WS_HIGH}) a=(${A_LOW},${A_HIGH}) b=(${B_LOW},${B_HIGH}) f=(${F_LOW},${F_HIGH})"
echo "  output root     : Exps/${EXP_DIR}/"
echo
printf "  %-8s %4s %4s %14s %8s %5s\n" CONFIG nb hsf params "%base" GPU
for c in "${CONFIGS[@]}"; do
    read -r ID NB HSF P G <<< "$c"
    printf "  %-8s %4s %4s %14s %7.1f%% %5s\n" "$ID" "$NB" "$HSF" "$P" \
        "$(awk -v a="$P" -v b="$BASELINE_PARAMS" 'BEGIN{print 100*a/b}')" "$G"
done
echo

if [[ ${DRY_RUN} -eq 1 ]]; then echo "  --dry_run: nothing launched."; exit 0; fi

run_one() {
    local ID=$1 NB=$2 HSF=$3 EXPECTED=$4 GPU=$5
    local NOTE="${ID}_nb${NB}_hsf${HSF}"
    local RUN_DIR="${REPO_ROOT}/Exps/${EXP_DIR}/${NOTE}"

    echo ">>> [$(date '+%F %T')] GPU ${GPU} | ${ID} | nb=${NB} hsf=${HSF} | expect ${EXPECTED} params"

    local COMMON=(
        --backbone "${BACKBONE}" --seed "${SEED}"
        --exp_dir "${EXP_DIR}" --exp_note "${NOTE}"
        --dataset "${DATASET}" --img_size "${IMG_SIZE}" --img_channel "${IMG_CHANNEL}"
        --seq_len "${SEQ_LEN}" --frames_in "${FRAMES_IN}" --frames_out "${FRAMES_OUT}"
        --num_workers "${NUM_WORKERS}"
        --wave "${WAVE}" --wavelet_level "${WAVELET_LEVEL}" --hf_mode "${HF_MODE}"
        --weight_scale_low "${WS_LOW}" --alpha_low "${A_LOW}"
        --beta_low "${B_LOW}" --freq_multiplier_low "${F_LOW}"
        --weight_scale_high "${WS_HIGH}" --alpha_high "${A_HIGH}"
        --beta_high "${B_HIGH}" --freq_multiplier_high "${F_HIGH}"
        --spectral_blocks "${NB}" --spectral_hidden_size_factor "${HSF}"
        --sparsity_threshold "${SPARSITY_THRESHOLD}" --conv_kernel "${CONV_KERNEL}"
        --hidden_dim "${HIDDEN_DIM}" --size_factor "${SIZE_FACTOR}"
        --wandb_state "${WANDB_STATE}" --wandb_project_name "${WANDB_PROJECT}"
        --run_name "meteo_${NOTE}" --gpu_use "${GPU}"
    )

    # ---- train + validate (earthformer) ----
    CUDA_VISIBLE_DEVICES=${GPU} conda run --no-capture-output -n "${TRAIN_ENV}" \
        python3 "${RUNNER}" "${COMMON[@]}" \
        --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --lr "${LR}" --valid
    local TRAIN_RC=$?
    echo ">>> [$(date '+%F %T')] ${ID} train exited rc=${TRAIN_RC}"

    if [[ ${TRAIN_RC} -ne 0 ]]; then
        conda run -n "${TRAIN_ENV}" python3 "${LOGGER}" append \
            --run_dir "Exps/${EXP_DIR}/${NOTE}" --dataset "${DATASET}" \
            --config_id "${ID}" --baseline_params "${BASELINE_PARAMS}" \
            --gabor_source "${GABOR_SRC}" --status "train_failed_rc${TRAIN_RC}" \
            --notes "nb=${NB} hsf=${HSF}; training failed" 2>&1 | tail -1
        return
    fi

    # ---- test (alphapre_manual) ----
    CUDA_VISIBLE_DEVICES=${GPU} conda run --no-capture-output -n "${TEST_ENV}" \
        python3 "${RUNNER}" "${COMMON[@]}" \
        --batch_size "${BATCH_SIZE}" --eval \
        --ckpt_milestone "${RUN_DIR}/checkpoints/ckpt-best.pt"
    echo ">>> [$(date '+%F %T')] ${ID} eval exited rc=$?"

    # ---- record ----
    conda run -n "${TRAIN_ENV}" python3 "${LOGGER}" append \
        --run_dir "Exps/${EXP_DIR}/${NOTE}" --dataset "${DATASET}" \
        --config_id "${ID}" --baseline_params "${BASELINE_PARAMS}" \
        --gabor_source "${GABOR_SRC}" --status "complete" \
        --notes "nb=${NB} hsf=${HSF} ratio=hsf/nb; reduction vs MET-B0" 2>&1 | tail -1
}

worker() {
    local GPU=$1
    for c in "${CONFIGS[@]}"; do
        read -r ID NB HSF P G <<< "$c"
        [[ "$G" == "$GPU" ]] && run_one "$ID" "$NB" "$HSF" "$P" "$G"
    done
    echo ">>> [$(date '+%F %T')] GPU ${GPU} worker done."
}

worker 0 &
worker 1 &
wait
echo "============================================================"
echo " Meteonet parameter-reduction sweep finished $(date '+%F %T')"
echo "============================================================"
