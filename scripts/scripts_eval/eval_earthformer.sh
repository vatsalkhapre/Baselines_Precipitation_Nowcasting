CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm.py \
    --backbone earthformer \
    --dataset meteo \
    --ckpt_milestone /home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Earthformer/earthformer_on_meteonet_2/checkpoints/ckpt-best.pt \
    --eval \
    --seq_len 25 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'offline'