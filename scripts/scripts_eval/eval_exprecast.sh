CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir meteonet \
    --exp_note exprecast_on_meteonet \
    --batch_size 16 \
    --backbone exPreCast \
    --dataset meteo \
    --seq_len 25 \
    --eval \
    --lr 1e-3 \
    --epochs 200 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "exprecast_meteonet" 



CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir shanghai \
    --exp_note exprecast_on_shanghai \
    --batch_size 16 \
    --backbone exPreCast \
    --dataset shanghai \
    --seq_len 25 \
    --eval \
    --lr 1e-3 \
    --epochs 200 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "exprecast_shanghai" 


CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note exprecast_on_cikm \
    --batch_size 16 \
    --backbone exPreCast \
    --dataset cikm \
    --seq_len 15 \
    --eval \
    --lr 1e-3 \
    --epochs 200 \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "exprecast_cikm" 

CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --exp_dir sevir \
    --exp_note exprecast_on_sevir \
    --batch_size 16 \
    --backbone exPreCast \
    --dataset sevir \
    --seq_len 25 \
    --eval \
    --lr 1e-3 \
    --epochs 100 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "exprecast_sevir" 