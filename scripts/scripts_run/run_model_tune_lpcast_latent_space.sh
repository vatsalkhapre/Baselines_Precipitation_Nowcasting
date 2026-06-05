GPU_ID=1
AE_CKPT="$AE_CKPT_DIR/autoencoder_checkpoint_32_SHANGHAI.pth"

run_exp () {

    HIDDEN_DIM=$1
    LIFT_DIMS="$2"
    PROJ_DIMS="$3"

    EXP_TAG="hd${HIDDEN_DIM}_lift$(echo $LIFT_DIMS | tr ' ' '-')_proj$(echo $PROJ_DIMS | tr ' ' '-')"

    echo "================================================="
    echo "Running: $EXP_TAG"
    echo "================================================="

    CUDA_VISIBLE_DEVICES=$GPU_ID python $RUNNER \
        --backbone LPCast \
        --dataset shanghai_lr_latent_32 \
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
        --exp_dir lpcast_shanghai_tuning \
        --exp_note $EXP_TAG \
        --run_name LPCast_$EXP_TAG \
        --wandb_project_name ACML \
        --wandb_state online \
        --gpu_use $GPU_ID \
        --valid
}

# ==========================================================
# Sweep
# ==========================================================
# Original LPCast
run_exp 64 "64 64 64" "64 64 4"

# Alpha-like
run_exp 64 "64 64 64" "64 64 64 4"

run_exp 64 "32 64 64" "64 64 32 4"

# Wider
run_exp 96 "96 96 96" "96 96 4"

# Wider + deeper
run_exp 96 "96 96 96" "96 96 96 4"

# Large
run_exp 128 "128 128 128" "128 128 4"

# Large + deeper
run_exp 128 "128 128 128" "128 128 128 4"