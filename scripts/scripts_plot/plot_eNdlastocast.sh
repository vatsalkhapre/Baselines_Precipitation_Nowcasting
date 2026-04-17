CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone e_lastocast_d_haar \
      --dataset shanghai \
      --hidden_dim 64 \
      --weight_scale 1.0 \
      --freq_multiplier 1.5 \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 10 \
      --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/Encoder_and_Decoder_from_WSNET/lastocast_on_shanghai_pixel_space/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 


# CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
#       --backbone e_lastocast_d \
#       --dataset cikm \
#       --eval \
#       --plot \
#       --seq_len 15 \
#       --frames_in 5 \
#       --frames_out 10 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/Encoder_and_Decoder_from_WSNET/lastocast_on_cikm_pixel_space/checkpoints/ckpt-best.pt \
#       --wandb_state 'offline' 

