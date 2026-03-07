CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir meteonet \
    --exp_note earthformer_on_meteonet \
    --batch_size 4 \
    --backbone earthformer \
    --dataset meteo \
    --seq_len 25 \
    --epochs 50 \
    --valid \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthformer_meteonet" 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note earthformer_on_shanghai \
    --batch_size 4 \
    --backbone earthformer \
    --dataset shanghai \
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "earthformer_shanghai" 