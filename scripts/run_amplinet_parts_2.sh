# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.23.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2.3.23.1" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_2.3.23.1'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_2.3.23.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_2.3.23.1" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --eval \
#     --num_workers 8 \
#     --wandb_state 'offline'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_4.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_4.1" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_4.1'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_4.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_4.1" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --eval \
#     --num_workers 8 \
#     --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.2'

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_2.3.23.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_4.1 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_4.1" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_4.1'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_4.1 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_4.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_4.2 \
    --dataset meteo_lr_latent_32 \
    --exp_dir meteo_lr_latent_32_model_parts \
    --exp_note "amplinet_latent_falfcl_only_4.2" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_4.2'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_4.2 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_4.2" \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --eval \
#     --num_workers 8 \
#     --wandb_state 'offline' 

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_3.1 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_3.1" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_3.1'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_3.2 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_3.2" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_3.2'

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
#     --backbone amplinet_latent_falfcl_only_3.3 \
#     --dataset meteo_lr_latent_32 \
#     --exp_dir meteo_lr_latent_32_model_parts \
#     --exp_note "amplinet_latent_falfcl_only_3.3" \
#     --epochs 50 \
#     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
#     --valid \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name 'amplinet_latent_falfcl_only_3.3'CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py     --backbone amplinet_latent_falfcl_only_2.3.23.1     --dataset meteo_lr_latent_32     --exp_dir meteo_lr_latent_32_model_parts     --exp_note amplinet_latent_falfcl_only_2.3.23.1     --epochs 50     --ae_ckpt_path /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth     --valid     --num_workers 8     --wandb_state 'online'     --wandb_project_name 'Alphapre'     --run_name 'amplinet_latent_falfcl_only_2.3.23.1'
