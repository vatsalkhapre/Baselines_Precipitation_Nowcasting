#!/bin/bash
# ==============================================================
# Self-driving job queue. One instance per GPU; runs its jobs
# strictly sequentially and NEVER needs an agent in the loop.
#
#   bash THE_GABOR/orchestrate.sh <chain-name>
#
# Properties:
#   * idempotent  — a job whose log already contains "[done]" is skipped,
#                   so re-running a chain after a reboot resumes correctly
#   * chaining    — waits for an already-running job to reach "[done]"
#                   before starting the next one on that GPU
#   * detached    — launch with nohup; survives Claude Code disconnects
# ==============================================================
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate earthformer
export WANDB_API_KEY="${WANDB_API_KEY:-}"
LOGD=THE_GABOR/logs/_runlogs; mkdir -p $LOGD
STATE=THE_GABOR/logs/_runlogs/orchestrator_$1.state

note() { echo "[$(date '+%F %T')] $*" | tee -a "$STATE"; }

# wait until an already-running job's log reports completion
wait_done() {
  local L=$LOGD/$1.log
  [ -f "$L" ] || return 0
  grep -q '^\[done\]' "$L" 2>/dev/null && return 0
  note "WAIT  $1 (already running; polling for [done])"
  while ! grep -q '^\[done\]' "$L" 2>/dev/null; do
    pgrep -f "run_name $1" >/dev/null 2>&1 || pgrep -f "$1" >/dev/null 2>&1 || {
      note "WARN  $1 no longer running and never wrote [done]"; return 1; }
    sleep 120
  done
  note "OK    $1 finished"
}

wait_file() {  # $1 = path, $2 = human label
  [ -f "$1" ] && return 0
  note "WAIT  $2 (file $1)"
  while [ ! -f "$1" ]; do sleep 60; done
  note "OK    $2 available"
}

wait_gpus_free() {   # $1 = space-separated GPU indices, $2 = MiB threshold
  local IDX="$1" THRESH="${2:-2000}"
  while :; do
    local busy=0
    for g in $IDX; do
      local used
      used=$(nvidia-smi --id=$g --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [ -z "$used" ] && used=999999
      [ "$used" -gt "$THRESH" ] && busy=1
    done
    [ $busy -eq 0 ] && break
    note "WAIT  GPUs [$IDX] not free yet (need <${THRESH}MiB each); re-checking in 5 min"
    sleep 300
  done
  note "OK    GPUs [$IDX] are free"
}

# run a job unless it is already finished
run_job() {
  local NAME=$1; shift
  local L=$LOGD/$NAME.log
  if grep -q '^\[done\]' "$L" 2>/dev/null; then note "SKIP  $NAME (already [done])"; return 0; fi
  note "START $NAME"
  "$@" > "$L" 2>&1
  local rc=$?
  if grep -q '^\[done\]' "$L" 2>/dev/null; then note "OK    $NAME"
  else
    # preserve the evidence: a later retry truncates $L, so keep a dated copy
    mkdir -p $LOGD/_failed
    cp "$L" "$LOGD/_failed/${NAME}.$(date +%Y%m%d_%H%M%S).log" 2>/dev/null
    note "FAIL  $NAME rc=$rc  (log archived under $LOGD/_failed/)"
  fi
  return 0     # never abort the chain on one failure
}

COMMON=(--seed 0 --epochs 50 --val_every_epochs 5 --limit_val_batches 200
        --batch_size 4 --num_workers 8 --lr 1e-4 --stride 13
        --hf_mode separate --hidden_dim 64 --gabor_probe_every_epochs 1
        --wandb_project ICLR26 --wandb_state online)

# per-dataset architecture settings
cfg_for() {
  case $1 in
    cikm)     echo "--wave db4 --wavelet_level 2 --afno_blocks 1 --afno_hidden_size_factor 1 --k_spatial 7";;
    meteo)    echo "--wave db6 --wavelet_level 1 --afno_blocks 4 --afno_hidden_size_factor 4 --k_spatial 3";;
    shanghai) echo "--wave db6 --wavelet_level 3 --afno_blocks 4 --afno_hidden_size_factor 3 --k_spatial 3";;
    sevir)    echo "--wave db6 --wavelet_level 2 --afno_blocks 4 --afno_hidden_size_factor 4 --k_spatial 3";;
  esac
}
CIKM_INIT=(--freq_multiplier_low 22.74 --freq_multiplier_high 95.56
           --weight_scale_low 0.1 --weight_scale_high 0.25)

stage2() {  # dataset gpu [extra...]
  local DS=$1 GPU=$2; shift 2
  run_job Stage2_pixel_${DS}_seed0 env CUDA_VISIBLE_DEVICES=$GPU python -m THE_GABOR.run_stage2_pixel \
    --dataset $DS --donor_run Stage1_pixel_${DS}_seed0 --run_name Stage2_pixel_${DS}_seed0 \
    --transfer gabor "${COMMON[@]}" $(cfg_for $DS) "$@"
}
ablate() {  # dataset gpu key [init...]
  local DS=$1 GPU=$2 AB=$3; shift 3
  run_job Ablation_pixel_${DS}_${AB}_seed0 env CUDA_VISIBLE_DEVICES=$GPU python -m THE_GABOR.run_stage2_pixel \
    --dataset $DS --ablation $AB --run_name Ablation_pixel_${DS}_${AB}_seed0 \
    "${COMMON[@]}" $(cfg_for $DS) "$@"
}
nostage1() { # dataset gpu  (item D: random gabor init, all else identical)
  local DS=$1 GPU=$2
  run_job NoStage1_pixel_${DS}_seed0 env CUDA_VISIBLE_DEVICES=$GPU python -m THE_GABOR.run_stage2_pixel \
    --dataset $DS --run_name NoStage1_pixel_${DS}_seed0 --transfer \
    "${COMMON[@]}" $(cfg_for $DS)
}
partf() {  # target-regime donor-regime  (Part F, SEVIR pixel, .205 GPU0+2)
  local TGT=$1 DON=$2
  local NAME=PartF_sevir_${TGT}_gaborinit_${DON}_seed0
  wait_gpus_free "0 2"
  run_job $NAME env CUDA_VISIBLE_DEVICES=0,2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m THE_GABOR.run_stage2_pixel \
    --dataset sevir --sevir_regime $TGT --run_name $NAME \
    --donor_run Gabor_pixel_SEVIR_${DON}_seed0 --transfer gabor \
    --multi_gpu --batch_size 8 --num_workers 16 \
    --seed 0 --epochs 50 --val_every_epochs 5 --limit_val_batches 200 \
    --lr 1e-4 --stride 13 --hf_mode separate --hidden_dim 64 \
    --gabor_probe_every_epochs 1 $(cfg_for sevir) \
    --wandb_project ICLR26 --wandb_state online
}

seedrun() { # dataset gpu seed  (item E)
  local DS=$1 GPU=$2 SD=$3
  run_job Stage1_pixel_${DS}_seed${SD} env CUDA_VISIBLE_DEVICES=$GPU python -m THE_GABOR.run_stage1_pixel \
    --dataset $DS --run_name Stage1_pixel_${DS}_seed${SD} --seed $SD \
    --epochs 50 --val_every_epochs 5 --limit_val_batches 200 --batch_size 4 \
    --num_workers 8 --lr 1e-4 --stride 13 --hf_mode separate --hidden_dim 64 \
    $(cfg_for $DS | sed 's/--afno[^ ]* [0-9]*//g;s/--k_spatial [0-9]*//g') \
    --wandb_project ICLR26 --wandb_state online
  run_job Stage2_pixel_${DS}_seed${SD} env CUDA_VISIBLE_DEVICES=$GPU python -m THE_GABOR.run_stage2_pixel \
    --dataset $DS --donor_run Stage1_pixel_${DS}_seed${SD} --run_name Stage2_pixel_${DS}_seed${SD} \
    --seed $SD --transfer gabor --epochs 50 --val_every_epochs 5 --limit_val_batches 200 \
    --batch_size 4 --num_workers 8 --lr 1e-4 --stride 13 --hf_mode separate --hidden_dim 64 \
    --gabor_probe_every_epochs 1 $(cfg_for $DS) --wandb_project ICLR26 --wandb_state online
}

note "=== chain $1 starting ==="
case "$1" in
  # ---------------- .88 ----------------
  m88_gpu0)   wait_done Stage2_pixel_meteo_seed0
              # Part C gate: pick the better MeteoNet model (new 2-stage vs the
              # previous best published in results_table.tex) and emit the Gabor
              # init the ablations must use. Ablations MUST NOT run before this.
              MENV=$LOGD/meteo_init.env
              if [ ! -f "$MENV" ]; then
                note "SELECT meteo best model (Part C gate)"
                CUDA_VISIBLE_DEVICES=0 python -m THE_GABOR.select_best_init \
                  --dataset meteo --run Stage2_pixel_meteo_seed0 --out "$MENV" \
                  >> $LOGD/select_meteo.log 2>&1 || note "WARN select failed; see select_meteo.log"
              fi
              [ -f "$MENV" ] && . "$MENV" || ABL_INIT=""
              note "meteo ablation init: $ABL_INIT"
              for AB in a_no_wavelet c_no_gabor f_no_srst g_no_wgtm; do ablate meteo 0 $AB $ABL_INIT; done ;;
  m88_gpu1)   wait_done Stage2_pixel_shanghai_seed0
              # the Part C gate runs on gpu0; wait for the init it writes so both
              # halves of the meteo ablation set use the SAME winning config
              wait_file $LOGD/meteo_init.env "Part C meteo selection"
              . $LOGD/meteo_init.env
              note "meteo ablation init: $ABL_INIT"
              for AB in b_shared_fat d_no_str e_no_spatial; do ablate meteo 1 $AB $ABL_INIT; done
              note "Shanghai seeds moved to .66 chain m66_shanghai; Part F takes .88" ;;
  # ---------------- .66 ----------------
  m66_gpu0)   wait_done Stage2_pixel_cikm_seed0
              nostage1 cikm 0
              for SD in 1 2; do seedrun cikm 0 $SD; done
              wait_done Stage2_pixel_cikm_seed3
              wait_done Stage2_pixel_cikm_seed4
              note "SELECT best CIKM across seeds 0-4 vs previous best"
              wait_gpus_free "0"        # sibling chains may still hold the cards
              CUDA_VISIBLE_DEVICES=0 python -m THE_GABOR.select_best_seed \
                --dataset cikm --runs Stage2_pixel_cikm_seed0 \
                Stage2_pixel_cikm_seed1 Stage2_pixel_cikm_seed2 \
                Stage2_pixel_cikm_seed3 Stage2_pixel_cikm_seed4 \
                >> $LOGD/select_cikm.log 2>&1 || note "WARN cikm selection failed" ;;
  m66_gpu1)   wait_done Ablation_pixel_cikm_a_no_wavelet_seed0
              for AB in c_no_gabor f_no_srst g_no_wgtm; do ablate cikm 1 $AB "${CIKM_INIT[@]}"; done ;;
  m66_gpu2)   wait_done Ablation_pixel_cikm_b_shared_fat_seed0
              for AB in d_no_str e_no_spatial; do ablate cikm 2 $AB "${CIKM_INIT[@]}"; done
              for SD in 3 4; do seedrun cikm 2 $SD; done ;;
  m88_partf)  # Part F halves that run on .88 (needs both GPU0 and GPU1 free)
              wait_gpus_free "0 1"
              for SPEC in "random storm" "random random"; do
                set -- $SPEC; TGT=$1; DON=$2
                NAME=PartF_sevir_${TGT}_gaborinit_${DON}_seed0
                run_job $NAME env CUDA_VISIBLE_DEVICES=0,1 \
                  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                  python -m THE_GABOR.run_stage2_pixel \
                  --dataset sevir --sevir_regime $TGT --run_name $NAME \
                  --donor_run Gabor_pixel_SEVIR_${DON}_seed0 --transfer gabor \
                  --multi_gpu --batch_size 8 --num_workers 16 \
                  --seed 0 --epochs 50 --val_every_epochs 5 --limit_val_batches 200 \
                  --lr 1e-4 --stride 13 --hf_mode separate --hidden_dim 64 \
                  --gabor_probe_every_epochs 1 $(cfg_for sevir) \
                  --wandb_project ICLR26 --wandb_state online
              done ;;

  # ---------------- .66 : item E Shanghai seeds (moved off .88 for Part F) ----
  m66_shanghai) for SD in 1 2 3 4; do seedrun shanghai 2 $SD; done
              wait_gpus_free "2"
              CUDA_VISIBLE_DEVICES=2 python -m THE_GABOR.select_best_seed \
                --dataset shanghai --runs Stage2_pixel_shanghai_seed0 \
                Stage2_pixel_shanghai_seed1 Stage2_pixel_shanghai_seed2 \
                Stage2_pixel_shanghai_seed3 Stage2_pixel_shanghai_seed4 \
                >> $LOGD/select_shanghai.log 2>&1 || note "WARN shanghai selection failed" ;;

  # ---------------- .205 (GPU0+GPU2 only; GPU1 forbidden) ----------------
  m205)       wait_done Stage1_pixel_sevir_seed0
              # ORDER (as agreed): SEVIR first, then MeteoNet, then Part F.
              # SEVIR needs BOTH GPU0 and GPU2, so nothing else may occupy GPU2
              # while we wait -- that is why item D no longer runs first.
              wait_gpus_free "0 2"
              run_job Stage2_pixel_sevir_seed0 env CUDA_VISIBLE_DEVICES=0,2 \
                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m THE_GABOR.run_stage2_pixel \
                --dataset sevir --donor_run Stage1_pixel_sevir_seed0 --run_name Stage2_pixel_sevir_seed0 \
                --transfer gabor --multi_gpu --batch_size 8 --num_workers 16 \
                --seed 0 --epochs 50 --val_every_epochs 5 --limit_val_batches 200 \
                --lr 1e-4 --stride 13 --hf_mode separate --hidden_dim 64 \
                --gabor_probe_every_epochs 1 $(cfg_for sevir) \
                --wandb_project ICLR26 --wandb_state online
              # item D: MeteoNet without Stage 1 (single GPU, after SEVIR)
              nostage1 meteo 2
              # Part F: DAWN-Cast on SEVIR storm/random, Gabor from the existing
              # pixel Stage-1 storm/random donors (4 combinations)
              partf storm  storm      # F.a
              partf storm  random ;;   # F.b   (F.c/F.d run on .88)
  *) note "unknown chain '$1'"; exit 1;;
esac
note "=== chain $1 COMPLETE ==="
