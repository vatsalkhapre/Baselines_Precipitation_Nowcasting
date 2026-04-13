CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note mau_on_cikm \
    --batch_size 8 \
    --backbone mau \
    --dataset cikm \
    --seq_len 15 \
    --epochs 70 \
    --valid \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "mau_cikm" 


CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note mau_on_cikm \
    --batch_size 8 \
    --backbone mau \
    --dataset cikm \
    --seq_len 15 \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' 

