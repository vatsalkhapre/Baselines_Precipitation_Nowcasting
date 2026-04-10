CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note traj_gru_on_cikm \
    --batch_size 8 \
    --backbone traj_gru \
    --dataset cikm \
    --seq_len 15 \
    --epochs 70 \
    --valid \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "traj_gru_cikm" 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note traj_gru_on_cikm \
    --batch_size 8 \
    --backbone traj_gru \
    --dataset cikm \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note trajgru_on_shanghai \
    --batch_size 4 \
    --backbone traj_gru \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "traj_gru_shanghai" 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note traj_gru_on_shanghai \
    --batch_size 4 \
    --backbone traj_gru \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "traj_gru_shanghai" 