#!/bin/bash
# ==============================================================================
# DAWNCast — PIXEL-SPACE transfer experiments, full sequence
#
#   Source model : SEVIR pixel-space DAWNCast (frozen except the chosen surface)
#   Targets      : MeteoNet + Shanghai, 128x128 single-channel radar
#   Results CSV  : /home/vatsal/Dataserver2/Neurips/csv_files/Transfer_runs_pixel.csv
#                  (one final TEST row per run; "Model Params (in M)" column holds
#                   the TRAINABLE count, "Why?" holds trainable/total + lr + frac)
#
#   Experiments (each on BOTH datasets):
#     0. zeroshot       no training at all — the transfer floor / baseline
#     1. liftproj       temporal + lifting + projection            423,009
#     2. normbias       temporal + norms + biases, WHOLE model     249,761
#     3. normbias_stem  temporal + norms + biases, srst EXCLUDED   160,161
#     4. liftprojonly   lifting + projection only, temporal frozen 264,513
#     5. <best> on 50% of the training set
#     6. <best> on 20% of the training set
#
#   "best" = highest mean test CSI across the two datasets among 1-4, computed
#   from the CSV after those runs finish. Nothing is hard-coded.
#
#   usage:  bash run_all_pixel_transfer_experiments.sh [gpu0] [gpu1] [gpu2]
#           defaults to GPUs 0 1 2. Pass the same index 3x to serialise on one GPU.
#
#   Runtime: ~12-16 h on 3 GPUs (10 runs at 50 epochs + 2 zero-shot).
# ==============================================================================
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)

GPU_A=${1:-0}
GPU_B=${2:-1}
GPU_C=${3:-2}

SWEEP="scripts/scripts_run/run_dawncast_transfer_sweep_pixel.sh"
CSV="/home/vatsal/Dataserver2/Neurips/csv_files/Transfer_runs_pixel.csv"
CKPT="/home/vatsal/Dataserver2/Neurips/DAWNCast_pixelspace/dawncast_sevir_pixel/checkpoints/ckpt-best.pt"
LOGDIR="${ROOT}/Exps/transfer_sweep_pixel/_logs"
mkdir -p "${LOGDIR}"

# ------------------------------------------------------------------ preflight --
echo "=============================================================="
echo " PREFLIGHT"
echo "=============================================================="
[ -f "${CKPT}" ] || { echo "FATAL: pretrained checkpoint missing: ${CKPT}"; exit 1; }
[ -f "${SWEEP}" ] || { echo "FATAL: sweep script missing: ${SWEEP}"; exit 1; }
[ -f "finetune_temporal_path_transfer_pixel.py" ] || { echo "FATAL: runner missing"; exit 1; }
bash -n "${SWEEP}" || { echo "FATAL: sweep script has a syntax error"; exit 1; }
python -c "import ast;ast.parse(open('finetune_temporal_path_transfer_pixel.py').read())" \
    || { echo "FATAL: runner has a syntax error"; exit 1; }
mkdir -p "$(dirname ${CSV})"
echo "checkpoint : ${CKPT}"
echo "csv        : ${CSV}"
echo "gpus       : ${GPU_A} ${GPU_B} ${GPU_C}"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader
echo

run () {  # run <dataset> <config> <gpu> <lr> <frac> <logname>
    local ds=$1 cfg=$2 gpu=$3 lr=$4 frac=$5 name=$6
    echo "[$(date +%H:%M:%S)] START ${name} (gpu ${gpu})"
    bash "${SWEEP}" "${ds}" "${cfg}" "${gpu}" "${lr}" "${frac}" \
        > "${LOGDIR}/${name}.log" 2>&1
    local rc=$?
    if [ ${rc} -ne 0 ]; then
        echo "[$(date +%H:%M:%S)] *** FAILED ${name} (exit ${rc}) — see ${LOGDIR}/${name}.log"
    else
        echo "[$(date +%H:%M:%S)] DONE  ${name}"
    fi
    return 0            # never abort the batch because one run failed
}

# ------------------------------------------------- stage 0: zero-shot floor --
echo "=============================================================="
echo " STAGE 0 — zero-shot baseline (no training, ~10 min)"
echo "=============================================================="
run meteo    zeroshot "${GPU_A}" 1e-4 1.0 zeroshot_meteo    &
run shanghai zeroshot "${GPU_B}" 1e-4 1.0 zeroshot_shanghai &
wait
echo

# --------------------------------------- stage 1: the four adaptation surfaces --
echo "=============================================================="
echo " STAGE 1 — experiments 1-4 on both datasets (8 runs)"
echo "=============================================================="
( run meteo liftproj      "${GPU_A}" 1e-4 1.0 liftproj_meteo
  run shanghai liftproj   "${GPU_A}" 1e-4 1.0 liftproj_shanghai ) &
( run meteo normbias      "${GPU_B}" 1e-4 1.0 normbias_meteo
  run shanghai normbias   "${GPU_B}" 1e-4 1.0 normbias_shanghai ) &
( run meteo normbias_stem "${GPU_C}" 1e-4 1.0 normbias_stem_meteo
  run shanghai normbias_stem "${GPU_C}" 1e-4 1.0 normbias_stem_shanghai
  run meteo liftprojonly  "${GPU_C}" 1e-4 1.0 liftprojonly_meteo
  run shanghai liftprojonly "${GPU_C}" 1e-4 1.0 liftprojonly_shanghai ) &
wait
echo

# ------------------------------------------- pick the winner from the CSV --
echo "=============================================================="
echo " SELECTING BEST SURFACE (mean test CSI over both datasets)"
echo "=============================================================="
BEST=$(python - "${CSV}" <<'PYEOF'
import csv, sys, collections
path = sys.argv[1]
scores = collections.defaultdict(dict)
for r in csv.DictReader(open(path)):
    note = r.get('Experiment Details', '')
    # final TEST rows only: LPIPS is populated by utils.metrics, not by
    # utils.metrics_valid, so it distinguishes test rows from validation rows.
    if not note.startswith('pixeltransfer_') or not r.get('LPIPS'):
        continue
    if '_frac' in note:                       # data-fraction runs are not candidates
        continue
    parts = note.split('_')                   # pixeltransfer_<cfg>_<dataset>_lr...
    lr_i = next(i for i, p in enumerate(parts) if p.startswith('lr'))
    cfg = '_'.join(parts[1:lr_i - 1])
    dataset = parts[lr_i - 1]
    if cfg in ('zeroshot',):
        continue
    try:
        scores[cfg][dataset] = float(r['CSI-M'])
    except (TypeError, ValueError):
        continue

ranked = []
for cfg, per in scores.items():
    if len(per) == 2:                          # require BOTH datasets
        ranked.append((sum(per.values()) / 2, cfg, per))
ranked.sort(reverse=True)

for mean, cfg, per in ranked:
    print(f"  {cfg:<16} mean CSI {mean:.4f}   " +
          "  ".join(f"{d}={v:.4f}" for d, v in sorted(per.items())), file=sys.stderr)
print(ranked[0][1] if ranked else "liftproj")
PYEOF
)
echo "BEST SURFACE = ${BEST}"
echo

# ----------------------------------- stages 2 & 3: 50% and 20% of the data --
echo "=============================================================="
echo " STAGE 2 — ${BEST} on 50% and 20% of the training set (4 runs)"
echo "=============================================================="
( run meteo    "${BEST}" "${GPU_A}" 1e-4 0.5 ${BEST}_meteo_frac50
  run meteo    "${BEST}" "${GPU_A}" 1e-4 0.2 ${BEST}_meteo_frac20 ) &
( run shanghai "${BEST}" "${GPU_B}" 1e-4 0.5 ${BEST}_shanghai_frac50
  run shanghai "${BEST}" "${GPU_B}" 1e-4 0.2 ${BEST}_shanghai_frac20 ) &
wait
echo

# --------------------------------------------------------------- summary --
echo "=============================================================="
echo " FINAL TEST RESULTS  (${CSV})"
echo "=============================================================="
python - "${CSV}" <<'PYEOF'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r.get('Experiment Details', '').startswith('pixeltransfer_') and r.get('LPIPS')]
hdr = f"{'run':<44}{'trainable':>12}{'CSI':>9}{'CSI4':>8}{'CSI16':>8}{'HSS':>8}{'SSIM':>8}{'MSE':>10}{'LPIPS':>8}"
print(hdr); print('-' * len(hdr))
def f(r, k):
    try: return float(r[k])
    except (TypeError, ValueError): return float('nan')
for r in rows:
    print(f"{r['Experiment Details']:<44}{r['Model Params (in M)']:>12}"
          f"{f(r,'CSI-M'):>9.4f}{f(r,'CSI-4'):>8.4f}{f(r,'CSI-16'):>8.4f}"
          f"{f(r,'HSS'):>8.4f}{f(r,'SSIM'):>8.4f}{f(r,'MSE'):>10.3f}{f(r,'LPIPS'):>8.4f}")
PYEOF
echo
echo "All pixel-space transfer experiments finished. Logs: ${LOGDIR}"
