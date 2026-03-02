CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone AFNO_ablation \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_ablations \
    --exp_note "AFNO_ablation" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --valid \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name AFNO_ablation_meteonet_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone AFNO_ablation \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_ablations \
    --exp_note "AFNO_ablation" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --eval \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone AFNO_ablation \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_ablations \
    --exp_note "AFNO_ablation" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --valid \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name AFNO_ablation_shanghai_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone AFNO_ablation \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_ablations \
    --exp_note "AFNO_ablation" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --eval \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone AFNO_ablation \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_ablations \
    --exp_note "AFNO_ablation" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name AFNO_ablation_CIKM_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone AFNO_ablation \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_ablations \
    --exp_note "AFNO_ablation" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone FNO_ablation \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_ablations \
    --exp_note "FNO_ablation" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name FNO_ablation_CIKM_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone FNO_ablation \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_ablations \
    --exp_note "FNO_ablation" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --eval \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 