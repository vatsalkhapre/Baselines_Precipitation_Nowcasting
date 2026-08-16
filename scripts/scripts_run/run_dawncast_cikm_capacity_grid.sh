#!/bin/bash
# ============================================================
# PART B -- DAWN-Cast CIKM capacity grid: pixel-space training launcher
#
# Runner   : run_alphapre_convlstm.py   (UNMODIFIED -- every knob used below
#                                        already exists as a CLI flag)
# Backbone : DAWNCast -> models/DAWNCast/dawncast.py   (UNMODIFIED)
# Space    : pixel (raw 128x128 CIKM frames, no autoencoder)
#
# Consumes scripts/cikm_valid_combos.csv produced by
#   python3 scripts/dawncast_cikm_grid_enumerate.py
# and trains + evaluates one DAWN-Cast per surviving (num_blocks,
# hidden_size_factor, level) combination.
#
# The eight Gabor init hyperparameters are REQUIRED arguments with no
# defaults -- the script refuses to run without all eight.
#
# Nothing launches without an explicit confirmation (--confirm, or an
# interactive "yes"). Use --dry_run to print the plan and collision report only.
#
# Usage:
#   scripts/scripts_run/run_dawncast_cikm_capacity_grid.sh \
#       --weight_scale_low 0.1  --weight_scale_high 0.25 \
#       --alpha_low 1.0         --alpha_high 1.0 \
#       --beta_low 43.1034      --beta_high 4.8193 \
#       --freq_multiplier_low 22.74 --freq_multiplier_high 3.04 \
#       --gpus 0,1,2 --dry_run
# ============================================================

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${REPO_ROOT}/run_alphapre_convlstm.py"
COMBOS_CSV="${REPO_ROOT}/scripts/cikm_valid_combos.csv"

# ---- Fixed for this study (per spec) -------------------------
BACKBONE="DAWNCast"
DATASET="cikm"
IMG_SIZE=128
IMG_CHANNEL=1
SEQ_LEN=15
FRAMES_IN=5          # CIKM T_in  (run_alphapre_convlstm.py cikm branch)
FRAMES_OUT=10        # CIKM T_out -- MUST be passed explicitly, see NOTE below
HIDDEN_DIM=64
HF_MODE="separate"
SIZE_FACTOR=1.0      # resolved default of get_model(size_factor=...)
CONV_KERNEL=3        # k_spatial

# NOTE: Runner.__init__ forces frames_in=5/frames_out=10 for cikm, but only
# AFTER _load_data() and _build_model() have already run. The model is therefore
# built from the CLI --frames_out. Passing --frames_out 10 explicitly is what
# makes hidden_size = 64*10 = 640, which is the value the Part A divisor
# enumeration and the 55M filter are based on.

# ---- Tunable, with stated defaults ---------------------------
EXP_DIR="cikm_dawncast_capacity_grid"   # -> Exps/<EXP_DIR>/<run_name>/
WAVE="db4"                              # matches run_freq_sweep_cikm.sh
SPARSITY_THRESHOLD=0.01
EPOCHS=50
SEED=0
BATCH_SIZE=4
LR=1e-4
NUM_WORKERS=8
GPUS="0"
WANDB_STATE="online"
WANDB_PROJECT="DAWNCast_CIKM_capacity_grid"

# ---- Gabor init: REQUIRED, no defaults -----------------------
WS_LOW=""; WS_HIGH=""
A_LOW="";  A_HIGH=""
B_LOW="";  B_HIGH=""
F_LOW="";  F_HIGH=""

CONFIRM=0
DRY_RUN=0
ALLOW_EXISTING=0

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}"
    cat <<'EOF'

Required (no defaults -- Gabor init):
  --weight_scale_low VAL    --weight_scale_high VAL
  --alpha_low VAL           --alpha_high VAL
  --beta_low VAL            --beta_high VAL
  --freq_multiplier_low VAL --freq_multiplier_high VAL

Optional:
  --combos_csv PATH   (default scripts/cikm_valid_combos.csv)
  --exp_dir NAME      (default cikm_dawncast_capacity_grid)
  --wave NAME         (default db4)
  --sparsity_threshold VAL (default 0.01)
  --epochs N --seed N --batch_size N --lr VAL --num_workers N
  --gpus 0,1,2        round-robin workers, run in parallel
  --wandb_state online|offline|disabled   --wandb_project NAME
  --dry_run           print plan + collision report, launch nothing
  --confirm           skip the interactive confirmation
  --allow_existing    proceed even if an output dir already exists
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --weight_scale_low)     WS_LOW="$2"; shift 2 ;;
        --weight_scale_high)    WS_HIGH="$2"; shift 2 ;;
        --alpha_low)            A_LOW="$2"; shift 2 ;;
        --alpha_high)           A_HIGH="$2"; shift 2 ;;
        --beta_low)             B_LOW="$2"; shift 2 ;;
        --beta_high)            B_HIGH="$2"; shift 2 ;;
        --freq_multiplier_low)  F_LOW="$2"; shift 2 ;;
        --freq_multiplier_high) F_HIGH="$2"; shift 2 ;;
        --combos_csv)           COMBOS_CSV="$2"; shift 2 ;;
        --exp_dir)              EXP_DIR="$2"; shift 2 ;;
        --wave)                 WAVE="$2"; shift 2 ;;
        --sparsity_threshold)   SPARSITY_THRESHOLD="$2"; shift 2 ;;
        --epochs)               EPOCHS="$2"; shift 2 ;;
        --seed)                 SEED="$2"; shift 2 ;;
        --batch_size)           BATCH_SIZE="$2"; shift 2 ;;
        --lr)                   LR="$2"; shift 2 ;;
        --num_workers)          NUM_WORKERS="$2"; shift 2 ;;
        --gpus)                 GPUS="$2"; shift 2 ;;
        --wandb_state)          WANDB_STATE="$2"; shift 2 ;;
        --wandb_project)        WANDB_PROJECT="$2"; shift 2 ;;
        --dry_run)              DRY_RUN=1; shift ;;
        --confirm)              CONFIRM=1; shift ;;
        --allow_existing)       ALLOW_EXISTING=1; shift ;;
        -h|--help)              usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'"; usage; exit 2 ;;
    esac
done

# ---- Enforce: Gabor init has no defaults ---------------------
MISSING=()
[[ -z "${WS_LOW}"  ]] && MISSING+=("--weight_scale_low")
[[ -z "${WS_HIGH}" ]] && MISSING+=("--weight_scale_high")
[[ -z "${A_LOW}"   ]] && MISSING+=("--alpha_low")
[[ -z "${A_HIGH}"  ]] && MISSING+=("--alpha_high")
[[ -z "${B_LOW}"   ]] && MISSING+=("--beta_low")
[[ -z "${B_HIGH}"  ]] && MISSING+=("--beta_high")
[[ -z "${F_LOW}"   ]] && MISSING+=("--freq_multiplier_low")
[[ -z "${F_HIGH}"  ]] && MISSING+=("--freq_multiplier_high")
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: these Gabor init arguments are required and have no defaults:"
    printf '    %s\n' "${MISSING[@]}"
    echo
    echo "Run with --help for usage."
    exit 2
fi

[[ -f "${COMBOS_CSV}" ]] || { echo "ERROR: combos CSV not found: ${COMBOS_CSV}"
    echo "Run Part A first:  python3 scripts/dawncast_cikm_grid_enumerate.py"; exit 2; }

# ============================================================
# Naming scheme  (printed before anything is touched)
# ============================================================
#   run name / exp_note / output dir :
#       cikm_pixel_dawncast_nb{num_blocks}_hsf{hsf}_lvl{level}
#   checkpoints : Exps/<EXP_DIR>/<name>/checkpoints/ckpt-{best,last}.pt
#   log file    : Exps/<EXP_DIR>/<name>/logs/log.log
#   params dump : Exps/<EXP_DIR>/<name>/params.yaml
#   results CSV : /home/vatsal/Dataserver2/Neurips/csv_files/Rebuttal_runs.csv
#                 (shared, append-only; row keyed by Model/Dataset/"Experiment
#                  Details"=<name>/"Model Params (in M)"; written on --eval only)
#   wandb run   : <name>_p{params}   in project <WANDB_PROJECT>
name_for() {  # $1=nb $2=hsf $3=level
    echo "cikm_pixel_dawncast_nb${1}_hsf${2}_lvl${3}"
}

mapfile -t COMBO_ROWS < <(tail -n +2 "${COMBOS_CSV}" | grep -v '^[[:space:]]*$')
N_COMBOS=${#COMBO_ROWS[@]}

echo "============================================================"
echo " PART B -- DAWN-Cast CIKM pixel-space capacity grid"
echo "============================================================"
echo "  runner            : ${RUNNER}"
echo "  backbone          : ${BACKBONE}   (dataset=${DATASET}, pixel space)"
echo "  combos CSV        : ${COMBOS_CSV}"
echo "  output root       : ${REPO_ROOT}/Exps/${EXP_DIR}/"
echo "  results CSV       : /home/vatsal/Dataserver2/Neurips/csv_files/Rebuttal_runs.csv (shared, on --eval)"
echo "  wandb             : project=${WANDB_PROJECT}  state=${WANDB_STATE}"
echo
echo "  naming scheme     : cikm_pixel_dawncast_nb{num_blocks}_hsf{hsf}_lvl{level}"
echo "                      (used for --exp_note, so it is also the checkpoint dir,"
echo "                       log dir and params.yaml dir; wandb run name appends _p{params})"
echo
echo "  fixed             : hidden_dim=${HIDDEN_DIM} hf_mode=${HF_MODE} size_factor=${SIZE_FACTOR} k_spatial=${CONV_KERNEL}"
echo "                      T_in=${FRAMES_IN} T_out=${FRAMES_OUT} img=${IMG_SIZE}x${IMG_SIZE} ch=${IMG_CHANNEL} seq_len=${SEQ_LEN}"
echo "                      wave=${WAVE} sparsity_threshold=${SPARSITY_THRESHOLD} epochs=${EPOCHS} seed=${SEED} bs=${BATCH_SIZE} lr=${LR}"
echo
echo "  Gabor init (supplied, not defaulted):"
echo "      weight_scale_low=${WS_LOW}        weight_scale_high=${WS_HIGH}"
echo "      alpha_low=${A_LOW}                alpha_high=${A_HIGH}"
echo "      beta_low=${B_LOW}                 beta_high=${B_HIGH}"
echo "      freq_multiplier_low=${F_LOW}      freq_multiplier_high=${F_HIGH}"
echo

# ---- Collision check: within-grid and against existing outputs ----
declare -A SEEN
DUPES=0
EXISTING=()
for row in "${COMBO_ROWS[@]}"; do
    IFS=',' read -r NB HSF LVL PARAMS <<< "${row}"
    NAME="$(name_for "${NB}" "${HSF}" "${LVL}")"
    if [[ -n "${SEEN[$NAME]:-}" ]]; then
        echo "  [COLLISION] duplicate run name within grid: ${NAME}"
        DUPES=$((DUPES + 1))
    fi
    SEEN[$NAME]=1
    [[ -e "${REPO_ROOT}/Exps/${EXP_DIR}/${NAME}" ]] && EXISTING+=("${NAME}")
done

echo "  collision report:"
echo "      unique run names / combos : ${#SEEN[@]} / ${N_COMBOS}  ($([[ ${DUPES} -eq 0 ]] && echo 'no within-grid collisions' || echo "${DUPES} DUPLICATES"))"
if [[ ${#EXISTING[@]} -gt 0 ]]; then
    echo "      pre-existing output dirs  : ${#EXISTING[@]}  <-- would be written into"
    printf '          %s\n' "${EXISTING[@]:0:10}"
    [[ ${#EXISTING[@]} -gt 10 ]] && echo "          ... and $(( ${#EXISTING[@]} - 10 )) more"
else
    echo "      pre-existing output dirs  : 0  (Exps/${EXP_DIR}/ is clean)"
fi
echo "      the 'cikm_pixel_dawncast_' prefix + a dedicated exp_dir keeps these"
echo "      clear of existing AlphaPre/ConvLSTM/DAWNCast_old outputs under Exps/."
echo

if [[ ${DUPES} -gt 0 ]]; then
    echo "ERROR: within-grid run-name collisions detected. Aborting."; exit 3
fi
if [[ ${#EXISTING[@]} -gt 0 && ${ALLOW_EXISTING} -eq 0 ]]; then
    echo "ERROR: ${#EXISTING[@]} output dir(s) already exist. Re-run with --allow_existing to proceed."
    exit 3
fi

# ============================================================
# Confirmation gate  -- each combo is a FULL training run
# ============================================================
echo "============================================================"
echo "  SURVIVING COMBOS TO TRAIN : ${N_COMBOS}"
echo "  Each is a full ${EPOCHS}-epoch training run + a test-set eval pass."
echo "  GPUs: ${GPUS}"
echo "============================================================"

if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  --dry_run set: nothing launched."
    exit 0
fi

if [[ ${CONFIRM} -eq 0 ]]; then
    if [[ ! -t 0 ]]; then
        echo "ERROR: no TTY for the interactive prompt and --confirm not given. Aborting."
        exit 4
    fi
    read -r -p "  Launch all ${N_COMBOS} training runs? Type 'yes' to proceed: " REPLY
    if [[ "${REPLY}" != "yes" ]]; then
        echo "  Aborted -- nothing launched."; exit 0
    fi
fi

# ============================================================
# Launch
# ============================================================
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
N_GPUS=${#GPU_ARR[@]}

run_one() {
    local GPU=$1 NB=$2 HSF=$3 LVL=$4 PARAMS=$5
    local NAME; NAME="$(name_for "${NB}" "${HSF}" "${LVL}")"
    local RUN_NAME="${NAME}_p${PARAMS}"

    echo "=== GPU ${GPU} | ${NAME} | ${PARAMS} params ==="

    local COMMON=(
        --backbone                      "${BACKBONE}"
        --seed                          "${SEED}"
        --exp_dir                       "${EXP_DIR}"
        --exp_note                      "${NAME}"
        --dataset                       "${DATASET}"
        --img_size                      "${IMG_SIZE}"
        --img_channel                   "${IMG_CHANNEL}"
        --seq_len                       "${SEQ_LEN}"
        --frames_in                     "${FRAMES_IN}"
        --frames_out                    "${FRAMES_OUT}"
        --num_workers                   "${NUM_WORKERS}"
        --wave                          "${WAVE}"
        --wavelet_level                 "${LVL}"
        --hf_mode                       "${HF_MODE}"
        --weight_scale_low              "${WS_LOW}"
        --alpha_low                     "${A_LOW}"
        --beta_low                      "${B_LOW}"
        --freq_multiplier_low           "${F_LOW}"
        --weight_scale_high             "${WS_HIGH}"
        --alpha_high                    "${A_HIGH}"
        --beta_high                     "${B_HIGH}"
        --freq_multiplier_high          "${F_HIGH}"
        --spectral_blocks               "${NB}"
        --spectral_hidden_size_factor   "${HSF}"
        --sparsity_threshold            "${SPARSITY_THRESHOLD}"
        --conv_kernel                   "${CONV_KERNEL}"
        --hidden_dim                    "${HIDDEN_DIM}"
        --size_factor                   "${SIZE_FACTOR}"
        --wandb_state                   "${WANDB_STATE}"
        --wandb_project_name            "${WANDB_PROJECT}"
        --run_name                      "${RUN_NAME}"
        --gpu_use                       "${GPU}"
    )

    # ---- train (valid every 5 epochs, keeps ckpt-best.pt) ----
    CUDA_VISIBLE_DEVICES=${GPU} python3 "${RUNNER}" \
        "${COMMON[@]}" \
        --epochs      "${EPOCHS}" \
        --batch_size  "${BATCH_SIZE}" \
        --lr          "${LR}" \
        --valid

    # ---- eval (writes the shared results CSV; ResultsLogger only fires
    #      under --eval, tagging the row with exp_note + model params) ----
    CUDA_VISIBLE_DEVICES=${GPU} python3 "${RUNNER}" \
        "${COMMON[@]}" \
        --batch_size  "${BATCH_SIZE}" \
        --eval
}

worker() {
    local WID=$1 GPU=${GPU_ARR[$1]}
    local idx=0
    for row in "${COMBO_ROWS[@]}"; do
        if [[ $(( idx % N_GPUS )) -eq ${WID} ]]; then
            IFS=',' read -r NB HSF LVL PARAMS <<< "${row}"
            run_one "${GPU}" "${NB}" "${HSF}" "${LVL}" "${PARAMS}"
        fi
        idx=$((idx + 1))
    done
}

for (( w=0; w<N_GPUS; w++ )); do
    worker "${w}" &
done
wait

echo "============================================================"
echo " All ${N_COMBOS} runs finished."
echo "============================================================"
