CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note e_lastocast_d_haar_on_cikm_pixel_space_skipcon \
    --batch_size 4 \
    --hidden_dim 64 \
    --size_factor 1.0 \
    --backbone e_lastocast_d_haar \
    --dataset cikm \
    --seq_len 15 \
    --valid \
    --weight_scale 1.0 \
    --freq_multiplier 1.5 \
    --epochs 100 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name "Newe&d_wsnet_haar_e_lastocast_d_haar_cikm_pixel_space_skipcon" 


CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note e_lastocast_d_haar_on_cikm_pixel_space_skipcon \
    --batch_size 4 \
    --hidden_dim 64 \
    --size_factor 1.0 \
    --backbone e_lastocast_d_haar \
    --dataset cikm \
    --seq_len 15 \
    --eval \
    --weight_scale 1.0 \
    --freq_multiplier 1.5 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 