#!/usr/bin/env bash
# =============================================================================
# WADEPre baseline (wavelet Approximation/Detail decomposition + Refiner).
#
# WADEPre is a same-length model: it is trained single-step T_in -> T_in
# (frames_in=5 -> next 5 frames) and, at inference, rolls out autoregressively
# ceil(T_out / T_in) times to produce the full forecast (4x for 5 -> 20). All
# of this is handled inside models/WADEPre/wadepre.py::predict, so the runner is
# used exactly like the other backbones.
#
# Architecture / loss hyperparameters (idr_dim=64, feature_channel=128,
# approx_hidden_size=256, refine_hidden_dim~576 [snapped to the nearest value
# valid for timesteps=T_in=5], wavelet=bior2.4 level 3, loss weights) follow
# the paper's reported config exactly and are baked into get_model, since the
# runner does not expose them as CLI flags.
#
# Optimizer / schedule (paper: AdamW, lr=1.5e-4, betas=(0.9, 0.995),
# weight_decay=0.01, fp32, CosineAnnealingLR with T_max=200) is applied via a
# wadepre-specific branch added to run_alphapre_convlstm.py::_build_optimizer
# -- --lr/--lr_beta1/--lr_beta2 below feed that branch directly, and
# --scheduler is ignored for this backbone (the runner always builds a plain
# no-warmup CosineAnnealingLR for wadepre). --epochs 200 matches the paper's
# T_max=200.
# =============================================================================

# ---------------------------- SEVIR (train + valid) --------------------------
CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir sevir \
    --exp_note wadepre_on_sevir \
    --batch_size 4 \
    --backbone wadepre \
    --dataset sevir \
    --seq_len 25 \
    --epochs 100 \
    --valid \
    --img_size 128 \
    --img_channel 1 \
    --frames_in 5 \
    --frames_out 20 \
    --lr 1.5e-4 \
    --lr_beta1 0.9 \
    --lr_beta2 0.995 \
    --scheduler cosine \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "wadepre_sevir"


# ---------------------------- Shanghai (train + valid) -----------------------
# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir shanghai \
#     --exp_note wadepre_on_shanghai \
#     --batch_size 4 \
#     --backbone wadepre \
#     --dataset shanghai \
#     --seq_len 25 \
#     --epochs 200 \
#     --valid \
#     --img_size 128 \
#     --img_channel 1 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --lr 1.5e-4 \
#     --lr_beta1 0.9 \
#     --lr_beta2 0.995 \
#     --scheduler cosine \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "wadepre_shanghai"


# ---------------------------- MeteoNet (train + valid) -----------------------
# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir meteonet \
#     --exp_note wadepre_on_meteonet \
#     --batch_size 4 \
#     --backbone wadepre \
#     --dataset meteo \
#     --seq_len 25 \
#     --epochs 200 \
#     --valid \
#     --img_size 128 \
#     --img_channel 1 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --lr 1.5e-4 \
#     --lr_beta1 0.9 \
#     --lr_beta2 0.995 \
#     --scheduler cosine \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "wadepre_meteonet"


# ---------------------------- CIKM (5 -> 10, 2x rollout) ---------------------
# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir cikm \
#     --exp_note wadepre_on_cikm \
#     --batch_size 4 \
#     --backbone wadepre \
#     --dataset cikm \
#     --seq_len 15 \
#     --epochs 200 \
#     --valid \
#     --img_size 128 \
#     --img_channel 1 \
#     --frames_in 5 \
#     --frames_out 10 \
#     --lr 1.5e-4 \
#     --lr_beta1 0.9 \
#     --lr_beta2 0.995 \
#     --scheduler cosine \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "wadepre_cikm"


# ---------------------------- Evaluation (from checkpoint) -------------------
# Point --ckpt_milestone at a saved ckpt-best.pt / ckpt-last.pt. With --eval and
# no --ckpt_milestone, the runner auto-resolves Exps/<exp_dir>/<exp_note>/checkpoints/ckpt-best.pt
# CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
#     --exp_dir sevir \
#     --exp_note wadepre_on_sevir \
#     --batch_size 4 \
#     --backbone wadepre \
#     --dataset sevir \
#     --seq_len 25 \
#     --eval \
#     --img_size 128 \
#     --img_channel 1 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/sevir/wadepre_on_sevir/checkpoints/ckpt-best.pt \
#     --wandb_state 'offline' \
#     --wandb_project_name 'Alphapre_all_other_models' \
#     --run_name "wadepre_sevir_eval"
