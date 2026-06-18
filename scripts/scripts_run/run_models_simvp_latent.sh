CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --exp_dir cikm_latent \
    --exp_note simvp \
    --batch_size 4 \
    --backbone simvp \
    --dataset meteo_lr_latent_32 \
    --seq_len 25 \
    --img_size 32 \
    --img_channel 4 \
    --valid \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name "Alphapre_meteo_latent_space" \
    --epochs 100 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --exp_dir meteo_latent \
    --exp_note simvp \
    --batch_size 4 \
    --backbone simvp \
    --dataset meteo_lr_latent_32 \
    --seq_len 25 \
    --img_size 32 \
    --img_channel 4 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre' \
    --run_name "Alphapre_meteo_latent_space" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --exp_dir cikm_latent \
    --exp_note simvp \
    --batch_size 4 \
    --backbone simvp \
    --dataset cikm_latent_32 \
    --seq_len 15 \
    --img_size 32 \
    --img_channel 4 \
    --valid \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name "Alphapre_cikm_latent_space" \
    --epochs 100 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_sevir_lr_latent.py \
    --exp_dir cikm_latent \
    --exp_note simvp \
    --batch_size 4 \
    --backbone simvp \
    --dataset cikm_latent_32 \
    --seq_len 15 \
    --img_size 32 \
    --img_channel 4 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre' \
    --run_name "Alphapre_latent_space" 

