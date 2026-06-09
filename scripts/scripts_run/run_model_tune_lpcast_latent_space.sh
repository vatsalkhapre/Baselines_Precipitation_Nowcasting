<<<<<<< HEAD
GPU_ID=1
AE_CKPT_DIR="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints"
AE_CKPT="$AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth"
DATASET='Sevir'
RUNNER=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent.py
run_exp () {
=======
# GPU_ID=1
# AE_CKPT_DIR="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints"
# RUNNER="run_alphapre_convlstm_sevir_lr_latent.py"

# AE_CKPT="$AE_CKPT_DIR/autoencoder_checkpoint_32_SHANGHAI.pth"
>>>>>>> a04adeb (sevir exps)

# run_exp () {

#     HIDDEN_DIM=$1
#     LIFT_DIMS="$2"
#     PROJ_DIMS="$3"

#     EXP_TAG="hd${HIDDEN_DIM}_lift$(echo $LIFT_DIMS | tr ' ' '-')_proj$(echo $PROJ_DIMS | tr ' ' '-')"

<<<<<<< HEAD
    CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
        --backbone LPCast \
        --dataset sevir_lr_latent_32 \
        --img_channel 4 \
        --img_size 32 \
        --frames_in 5 \
        --frames_out 20 \
        --seq_len 25 \
        --hidden_dim $HIDDEN_DIM \
        --mlp_size_factor 1.0 \
        --lift_dims $LIFT_DIMS \
        --proj_dims $PROJ_DIMS \
        --facl_const_ratio 0.1 \
        --ae_ckpt_path $AE_CKPT \
        --epochs 50 \
        --batch_size 4 \
        --num_workers 8 \
        --exp_dir lpcast_sevir_tuning \
        --exp_note $EXP_TAG \
        --run_name LPCast_$DATASET_$EXP_TAG \
        --wandb_project_name ACML \
        --wandb_state online \
        --gpu_use $GPU_ID \
        --valid
=======
#     echo "================================================="
#     echo "Running: $EXP_TAG"
#     echo "================================================="

#     CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
#         --backbone LPCast \
#         --dataset shanghai_lr_latent_32 \
#         --img_channel 4 \
#         --img_size 32 \
#         --frames_in 5 \
#         --frames_out 20 \
#         --seq_len 25 \
#         --hidden_dim $HIDDEN_DIM \
#         --mlp_size_factor 1.0 \
#         --lift_dims $LIFT_DIMS \
#         --proj_dims $PROJ_DIMS \
#         --facl_const_ratio 0.1 \
#         --ae_ckpt_path $AE_CKPT \
#         --epochs 50 \
#         --batch_size 4 \
#         --num_workers 8 \
#         --exp_dir lpcast_shanghai_tuning \
#         --exp_note $EXP_TAG \
#         --run_name LPCast_$EXP_TAG \
#         --wandb_project_name ACML \
#         --wandb_state offline \
#         --gpu_use $GPU_ID \
#         --eval
# }

# # ==========================================================
# # Sweep
# # ==========================================================
# # Original LPCast
# run_exp 64 "64 64 64" "64 64 4"

# # Alpha-like
# run_exp 64 "64 64 64" "64 64 64 4"

# run_exp 64 "32 64 64" "64 64 32 4"

# # Wider
# run_exp 96 "96 96 96" "96 96 4"

# # Wider + deeper
# run_exp 96 "96 96 96" "96 96 96 4"

# # Large
# run_exp 128 "128 128 128" "128 128 4"

# # Large + deeper
# run_exp 128 "128 128 128" "128 128 128 4"



#!/bin/bash

AE_CKPT_DIR="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints"
AE_CKPT="$AE_CKPT_DIR/autoencoder_checkpoint_32_SEVIR.pth"
DATASET="Sevir"
RUNNER="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/run_alphapre_convlstm_sevir_lr_latent.py"

# ── GPU Pool: named-pipe token bucket ─────────────────────────────────────
# Change these to your actual GPU IDs (e.g. (1 2 3) if GPU 0 is occupied)
GPU_POOL=(0 1 2)

_pipe=$(mktemp -u)
mkfifo "$_pipe"
exec 3<>"$_pipe"          # fd 3: bidirectional handle kept alive by the fd
rm   "$_pipe"             # unlink the name; the fd holds the pipe open
for _g in "${GPU_POOL[@]}"; do echo "$_g" >&3; done   # seed one token per GPU
# ──────────────────────────────────────────────────────────────────────────

run_exp() {
    local HIDDEN_DIM=$1
    local LIFT_DIMS="$2"
    local PROJ_DIMS="$3"
    local EXP_TAG="hd${HIDDEN_DIM}_lift$(echo "$LIFT_DIMS" | tr ' ' '-')_proj$(echo "$PROJ_DIMS" | tr ' ' '-')"

    (   # each experiment runs in its own subshell, launched in the background
        # Block here until a GPU token is available, then claim it
        read -u3 GPU_ID

        echo "================================================="
        echo "GPU $GPU_ID ▶  $EXP_TAG"
        echo "================================================="

        CUDA_VISIBLE_DEVICES=$GPU_ID python "$RUNNER" \
            --backbone LPCast_mse \
            --dataset sevir_lr_latent_32 \
            --img_channel 4 \
            --img_size 32 \
            --frames_in 5 \
            --frames_out 20 \
            --seq_len 25 \
            --hidden_dim $HIDDEN_DIM \
            --mlp_size_factor 1.0 \
            --lift_dims $LIFT_DIMS \
            --proj_dims $PROJ_DIMS \
            --facl_const_ratio 0.1 \
            --ae_ckpt_path "$AE_CKPT" \
            --epochs 50 \
            --batch_size 4 \
            --num_workers 8 \
            --exp_dir lpcast_mse_sevir_tuning \
            --exp_note "$EXP_TAG" \
            --run_name "lpcast_mse_${DATASET}_${EXP_TAG}" \
            --wandb_project_name ACML \
            --wandb_state offline \
            --gpu_use "$GPU_ID" \
            --eval
        EXIT_CODE=$?

        # Return the GPU token to the pool (always — even if the job failed)
        echo "$GPU_ID" >&3

        [[ $EXIT_CODE -eq 0 ]] \
            && echo "GPU $GPU_ID ✓ done:   $EXP_TAG" \
            || echo "GPU $GPU_ID ✗ FAILED (exit $EXIT_CODE): $EXP_TAG"
    ) &
>>>>>>> a04adeb (sevir exps)
}

# ==========================================================
# Sweep — 7 experiments queued across ${#GPU_POOL[@]} GPUs
# ==========================================================
<<<<<<< HEAD
run_exp 64  "64 64 64"    "64 64 4"        # Original LPCast
run_exp 64  "64 64 64"    "64 64 64 4"     # Alpha-like (deeper proj)
run_exp 64  "32 64 64"    "64 64 32 4"     # Alpha-like (narrower lift)
# run_exp 96  "96 96 96"    "96 96 4"        # Wider
# run_exp 96  "96 96 96"    "96 96 96 4"     # Wider + deeper
# run_exp 128 "128 128 128" "128 128 4"      # Large
# run_exp 128 "128 128 128" "128 128 128 4"  # Large + deeper
=======
# run_exp 64  "64 64 64"    "64 64 4"        # Original LPCast
# run_exp 64  "64 64 64"    "64 64 64 4"     # Alpha-like (deeper proj)
# run_exp 64  "32 64 64"    "64 64 32 4"     # Alpha-like (narrower lift)
run_exp 96  "96 96 96"    "96 96 4"        # Wider
run_exp 96  "96 96 96"    "96 96 96 4"     # Wider + deeper
run_exp 128 "128 128 128" "128 128 4"      # Large
run_exp 128 "128 128 128" "128 128 128 4"  # Large + deeper
>>>>>>> 2e181f6 (sevir tunning lpcast)

echo "All 7 experiments queued — waiting for completion…"
wait
exec 3>&-    # close the pipe
echo "All done."