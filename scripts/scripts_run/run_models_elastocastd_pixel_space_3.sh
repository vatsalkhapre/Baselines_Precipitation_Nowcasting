CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note lastocast_on_shanghai_pixel_space \
    --batch_size 4 \
    --hidden_dim 64 \
    --size_factor 1.0 \
    --backbone e_lastocast_d_haar \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --weight_scale 1.0 \
    --freq_multiplier 1.5 \
    --epochs 100 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name "Newe&d_wsnet_haar_lastocast_shanghai_pixel_space" 


CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note lastocast_on_shanghai_pixel_space \
    --batch_size 4 \
    --hidden_dim 64 \
    --size_factor 1.0 \
    --backbone e_lastocast_d_haar \
    --dataset shanghai \
    --seq_len 25 \
    --eval \
    --weight_scale 1.0 \
    --freq_multiplier 1.5 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' 