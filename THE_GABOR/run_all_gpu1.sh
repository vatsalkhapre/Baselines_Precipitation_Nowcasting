#!/bin/bash
# ==============================================================
# THE_GABOR -- Experiment 1
#   "Does Gabor learn differently when trained on different
#    precipitation regimes?"
#
# Runs one experiment at a time on GPU 1 (never concurrently).
# Every run starts from the SAME initial checkpoint for its space
# (created once per space+seed by THE_GABOR/make_init.py).
#
# Sequence:
#   1. Pixel  SEVIR RANDOM
#   2. Pixel  SEVIR STORM
#   3. Latent SEVIR RANDOM
#   4. Latent SEVIR STORM
#
# (Latent regime filtering IS supported: the latent CATALOG.csv keeps
#  the original file_name column, so RANDOM/STORM membership is
#  available exactly as in pixel space.  Use REGIMES_LATENT="all" to
#  run the standard, unfiltered latent SEVIR experiment instead.)
# ==============================================================
set -euo pipefail

cd "$(dirname "$0")/.."          # repository root

export CUDA_VISIBLE_DEVICES=1    # GPU 1, one run at a time

# ---- seeds ---------------------------------------------------
SEEDS=${SEEDS:-"0"}              # e.g. SEEDS="0 1 2"

# ---- W&B -----------------------------------------------------
WANDB_PROJECT=${WANDB_PROJECT:-"THE_GABOR"}
WANDB_STATE=${WANDB_STATE:-"online"}          # online | offline | disabled

# ---- controlled architecture (identical for every run) -------
WAVE="db6"
WAVELET_LEVEL=2                  # LL + HF_level_1 + HF_level_2
HF_MODE="separate"               # one Gabor per HF level
HIDDEN_DIM=64
SIZE_FACTOR=1.0

# ---- Gabor initialisation (no regime prior anywhere) ---------
FREQ_MULTIPLIER=1.0              # same for LL and every HF subband
WEIGHT_SCALE=0.1
ALPHA=1.0
BETA=1.0

# ---- pixel SEVIR ---------------------------------------------
PIX_IMG_SIZE=128
PIX_IMG_CHANNEL=1
PIX_FRAMES_IN=5                  # fixed by the experiment
PIX_FRAMES_OUT=20                # fixed by the experiment
PIX_SEQ_LEN=25
PIX_STRIDE=13
PIX_BATCH_SIZE=4
PIX_NUM_WORKERS=8
PIX_EPOCHS=50
PIX_LR=1e-4
# Caps steps/epoch.  NOTE: this does NOT equalise the two regimes.
# At seq_len=25/bs=4 RANDOM has 4710 batches/epoch and STORM only
# 1242, so the cap binds for RANDOM (-> 2000) but never for STORM
# (-> 1242).  Actual totals over 50 epochs: RANDOM 100,000 steps,
# STORM 62,100.  Read RANDOM-vs-STORM trajectories on the STEP axis,
# not the epoch axis.  Set to 0 to use every batch instead.
PIX_TRAIN_BATCHES=2000

# ---- latent SEVIR (sevir_lr_latent_32) -----------------------
LAT_IMG_SIZE=32
LAT_IMG_CHANNEL=4
LAT_FRAMES_IN=5
LAT_FRAMES_OUT=20                # matches pixel -- the two arms differ only in space
LAT_SEQ_LEN=25                   # matches pixel (latent HDF5 holds 49 frames/event)
LAT_STRIDE=13
LAT_BATCH_SIZE=4
LAT_NUM_WORKERS=8
LAT_EPOCHS=50
LAT_LR=1e-4
LAT_TRAIN_BATCHES=2000           # same flag value as pixel, so each latent regime
                                 # gets exactly the same steps/epoch as its pixel
                                 # counterpart: RANDOM 2000, STORM 1242 (capped by
                                 # its own dataset size, not by this flag)
AE_CKPT="Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SEVIR.pth"

# ---- logging cadence -----------------------------------------
GABOR_SCALAR_EVERY_STEPS=100
GABOR_PROBE_EVERY_EPOCHS=1
GABOR_HIST_EVERY_EPOCHS=5
PROBE_NEURONS=4
VAL_EVERY_EPOCHS=5
VAL_BATCHES=200

REGIMES_PIXEL=${REGIMES_PIXEL-"random storm"}   # export REGIMES_PIXEL="" to skip pixel
REGIMES_LATENT=${REGIMES_LATENT-"random storm"} # export REGIMES_LATENT="" to skip latent

# ==============================================================
if [ "${SKIP_SANITY:-0}" != "1" ]; then
    echo "=============== SANITY CHECKS ==============="
    python -m THE_GABOR.sanity_check
fi

for SEED in ${SEEDS}; do

    # ---------- ONE initial checkpoint per space + seed ----------
    echo "=============== INITIAL CHECKPOINTS (seed ${SEED}) ==============="
    python -m THE_GABOR.make_init --space pixel  --seed "${SEED}" \
        --frames_in ${PIX_FRAMES_IN} --frames_out ${PIX_FRAMES_OUT} \
        --img_channel ${PIX_IMG_CHANNEL} --hidden_dim ${HIDDEN_DIM} \
        --wave ${WAVE} --wavelet_level ${WAVELET_LEVEL} --hf_mode ${HF_MODE} \
        --freq_multiplier ${FREQ_MULTIPLIER} --weight_scale ${WEIGHT_SCALE} \
        --alpha ${ALPHA} --beta ${BETA} --size_factor ${SIZE_FACTOR}

    python -m THE_GABOR.make_init --space latent --seed "${SEED}" \
        --frames_in ${LAT_FRAMES_IN} --frames_out ${LAT_FRAMES_OUT} \
        --img_channel ${LAT_IMG_CHANNEL} --hidden_dim ${HIDDEN_DIM} \
        --wave ${WAVE} --wavelet_level ${WAVELET_LEVEL} --hf_mode ${HF_MODE} \
        --freq_multiplier ${FREQ_MULTIPLIER} --weight_scale ${WEIGHT_SCALE} \
        --alpha ${ALPHA} --beta ${BETA} --size_factor ${SIZE_FACTOR}

    # ---------- 1 + 2 : pixel SEVIR ----------
    for REGIME in ${REGIMES_PIXEL}; do
        RUN_NAME="Gabor_pixel_SEVIR_${REGIME}_seed${SEED}"
        echo "=============== ${RUN_NAME} (GPU 1) ==============="
        python -m THE_GABOR.run_pixel \
            --regime                  "${REGIME}" \
            --seed                    "${SEED}" \
            --run_name                "${RUN_NAME}" \
            --img_size                ${PIX_IMG_SIZE} \
            --img_channel             ${PIX_IMG_CHANNEL} \
            --frames_in               ${PIX_FRAMES_IN} \
            --frames_out              ${PIX_FRAMES_OUT} \
            --seq_len                 ${PIX_SEQ_LEN} \
            --stride                  ${PIX_STRIDE} \
            --batch_size              ${PIX_BATCH_SIZE} \
            --num_workers             ${PIX_NUM_WORKERS} \
            --epochs                  ${PIX_EPOCHS} \
            --lr                      ${PIX_LR} \
            --limit_train_batches     ${PIX_TRAIN_BATCHES} \
            --limit_val_batches       ${VAL_BATCHES} \
            --val_every_epochs        ${VAL_EVERY_EPOCHS} \
            --hidden_dim              ${HIDDEN_DIM} \
            --wave                    ${WAVE} \
            --wavelet_level           ${WAVELET_LEVEL} \
            --hf_mode                 ${HF_MODE} \
            --size_factor             ${SIZE_FACTOR} \
            --freq_multiplier         ${FREQ_MULTIPLIER} \
            --weight_scale            ${WEIGHT_SCALE} \
            --alpha                   ${ALPHA} \
            --beta                    ${BETA} \
            --gabor_scalar_every_steps ${GABOR_SCALAR_EVERY_STEPS} \
            --gabor_probe_every_epochs ${GABOR_PROBE_EVERY_EPOCHS} \
            --gabor_hist_every_epochs  ${GABOR_HIST_EVERY_EPOCHS} \
            --probe_neurons           ${PROBE_NEURONS} \
            --wandb_project           "${WANDB_PROJECT}" \
            --wandb_state             "${WANDB_STATE}"
    done

    # ---------- 3 + 4 : latent SEVIR ----------
    for REGIME in ${REGIMES_LATENT}; do
        RUN_NAME="Gabor_latent_SEVIR_${REGIME}_seed${SEED}"
        echo "=============== ${RUN_NAME} (GPU 1) ==============="
        python -m THE_GABOR.run_latent \
            --regime                  "${REGIME}" \
            --seed                    "${SEED}" \
            --run_name                "${RUN_NAME}" \
            --img_size                ${LAT_IMG_SIZE} \
            --img_channel             ${LAT_IMG_CHANNEL} \
            --frames_in               ${LAT_FRAMES_IN} \
            --frames_out              ${LAT_FRAMES_OUT} \
            --seq_len                 ${LAT_SEQ_LEN} \
            --stride                  ${LAT_STRIDE} \
            --batch_size              ${LAT_BATCH_SIZE} \
            --num_workers             ${LAT_NUM_WORKERS} \
            --epochs                  ${LAT_EPOCHS} \
            --lr                      ${LAT_LR} \
            --limit_train_batches     ${LAT_TRAIN_BATCHES} \
            --limit_val_batches       ${VAL_BATCHES} \
            --val_every_epochs        ${VAL_EVERY_EPOCHS} \
            --hidden_dim              ${HIDDEN_DIM} \
            --wave                    ${WAVE} \
            --wavelet_level           ${WAVELET_LEVEL} \
            --hf_mode                 ${HF_MODE} \
            --size_factor             ${SIZE_FACTOR} \
            --freq_multiplier         ${FREQ_MULTIPLIER} \
            --weight_scale            ${WEIGHT_SCALE} \
            --alpha                   ${ALPHA} \
            --beta                    ${BETA} \
            --gabor_scalar_every_steps ${GABOR_SCALAR_EVERY_STEPS} \
            --gabor_probe_every_epochs ${GABOR_PROBE_EVERY_EPOCHS} \
            --gabor_hist_every_epochs  ${GABOR_HIST_EVERY_EPOCHS} \
            --probe_neurons           ${PROBE_NEURONS} \
            --ae_ckpt_path            "${AE_CKPT}" \
            --wandb_project           "${WANDB_PROJECT}" \
            --wandb_state             "${WANDB_STATE}"
    done
done

echo "All THE_GABOR Experiment 1 runs finished (GPU 1)."
