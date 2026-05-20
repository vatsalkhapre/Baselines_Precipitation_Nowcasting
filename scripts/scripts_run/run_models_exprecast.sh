CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note exprecast_on_shanghai \
    --batch_size 4 \
    --backbone exPreCast \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "exprecast_shanghai" 

