CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteo_lr_latent_32_model_parts \
                --exp_note "amplinet_latent_falfcl_only_meteonet_2_3_13_2_3Dconv_corrected" \
                --epochs 50 \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                --valid \
                --seq_len 25 \
                --falfcl_weight 1 \
                --frames_in 5 \
                --frames_out 20 \
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_3Dconv_meteonet_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
                --dataset meteo_lr_latent_32 \
                --exp_dir meteo_lr_latent_32_model_parts \
                --exp_note "amplinet_latent_falfcl_only_meteonet_2_3_13_2_3Dconv_corrected" \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                --eval \
                --seq_len 25 \
                --falfcl_weight 1 \
                --frames_in 5 \
                --frames_out 20 \
                --num_workers 8 \
                --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
                --dataset shanghai_lr_latent_32 \
                --exp_dir shanghai_lr_latent_32_model_parts \
                --exp_note "amplinet_latent_falfcl_only_shanghai_2_3_13_2_3Dconv_corrected" \
                --epochs 50 \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
                --valid \
                --seq_len 25 \
                --falfcl_weight 1 \
                --frames_in 5 \
                --frames_out 20 \
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_3Dconv_shanghai_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
                --dataset shanghai_lr_latent_32 \
                --exp_dir shanghai_lr_latent_32_model_parts \
                --exp_note "amplinet_latent_falfcl_only_shanghai_2_3_13_2_3Dconv_corrected" \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
                --eval \
                --seq_len 25 \
                --falfcl_weight 1 \
                --frames_in 5 \
                --frames_out 20 \
                --num_workers 8 \
                --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
                --dataset cikm_latent_32 \
                --exp_dir cikm_latent_32_model_parts \
                --exp_note "amplinet_latent_falfcl_only_shanghai_2_3_13_2_3Dconv_corrected" \
                --epochs 50 \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
                --valid \
                --seq_len 15 \
                --falfcl_weight 1 \
                --frames_in 5 \
                --frames_out 10 \
                --num_workers 8 \
                --wandb_state 'online' \
                --wandb_project_name 'Alphapre' \
                --run_name amplinet_latent_falfcl_only_2_3_13_2_3Dconv_cikm_${weight_scale}_${a}_${b}_${f}

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
                --backbone amplinet_latent_falfcl_only_2_3_13_2_3Dconv \
                --dataset cikm_latent_32 \
                --exp_dir cikm_latent_32_model_parts \
                --exp_note "amplinet_latent_falfcl_only_shanghai_2_3_13_2_3Dconv_corrected" \
                --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
                --eval \
                --seq_len 15 \
                --falfcl_weight 1 \
                --frames_in 5 \
                --frames_out 10 \
                --num_workers 8 \
                --wandb_state 'offline' 