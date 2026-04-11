

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note fourcastnet_on_shanghai \
    --batch_size 4 \
    --backbone fourcastnet \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "fourcastnet_shanghai" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note fourcastnet_on_shanghai \
    --batch_size 4 \
    --backbone fourcastnet \
    --dataset shanghai \
    --seq_len 25 \
    --eval \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "fourcastnet_shanghai" 