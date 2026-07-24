CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir cikm \
    --exp_note lastocast_on_meteonet_pixel_space \
    --batch_size 4 \
    --backbone alphapre \
    --dataset cikm \
    --seq_len 15 \
    --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Alphapre_other_datasets/cikm_training_corr/checkpoints/ckpt-best.pt \
    --eval \
    --frames_in 5 \
    --frames_out 10 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
    --run_name "lastocast_meteonet_pixel_space" 


CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
    --exp_dir sevir \
    --exp_note alphapre_facl_loss \
    --batch_size 4 \
    --backbone alphapre_falfcl \
    --dataset sevir \
    --seq_len 25 \
    --epochs 50 \
    --valid \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre_all_other_models' \
    --run_name "alphapre_facl_sevir" 