# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
#     --exp_dir sevir \
#     --exp_note dawncast_on_sevir_pixel_space \
#     --batch_size 4 \
#     --backbone dawncast \
#     --dataset sevir \
#     --seq_len 25 \
#     --valid \
#     --epochs 50 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name "dawncast_sevir_pixel_space" 

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
#     --exp_dir shanghai \
#     --exp_note dawncast_on_shanghai_pixel_space \
#     --batch_size 4 \
#     --backbone dawncast \
#     --dataset shanghai \
#     --seq_len 25 \
#     --valid \
#     --epochs 50 \
#     --frames_in 5 \
#     --frames_out 20 \
#     --num_workers 8 \
#     --wandb_state 'online' \
#     --wandb_project_name 'Alphapre' \
#     --run_name "dawncast_shanghai_pixel_space" 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note dawncast_on_cikm_pixel_space \
    --batch_size 4 \
    --backbone dawncast \
    --dataset cikm \
    --seq_len 15 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name "dawncast_cikm_pixel_space" 