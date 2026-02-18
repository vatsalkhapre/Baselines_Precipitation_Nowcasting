CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3" \
    --epochs 50 \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3_cikm'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3_shanghai'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.1 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.1" \
    --epochs 50 \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3.1_cikm'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.1 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.1 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.1" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3.1_shanghai'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.1 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2" \
    --epochs 50 \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3.2_cikm'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3.2_shanghai'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2.1 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2.1" \
    --epochs 50 \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3.2.1_cikm'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2.1 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2.1 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2.1" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.3.2.1_shanghai'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.3.2.1 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.3.2.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2" \
    --epochs 50 \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.2_cikm'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.13.2_shanghai'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.13.2 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.13.2" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.23.1 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.23.1" \
    --epochs 50 \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.23.1_cikm'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.23.1 \
    --dataset cikm_latent_32 \
    --exp_dir cikm_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.23.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
    --seq_len 15 \
    --frames_in 5 \
    --frames_out 10 \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.23.1 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.23.1" \
    --epochs 50 \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --valid \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name 'amplinet_latent_falfcl_only_2.3.23.1_shanghai'

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --backbone amplinet_latent_falfcl_only_2.3.23.1 \
    --dataset shanghai_lr_latent_32 \
    --exp_dir shanghai_lr_latent_32_best_model \
    --exp_note "amplinet_latent_falfcl_only_2.3.23.1" \
    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
    --eval \
    --num_workers 8 \
    --wandb_state 'offline'