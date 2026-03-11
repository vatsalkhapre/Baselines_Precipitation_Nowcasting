CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --backbone alphapre \
    --dataset sevir \
    --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Alphapre_sevir/checkpoints/AlphaPre_sevir128.pt \
    --eval \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline'
