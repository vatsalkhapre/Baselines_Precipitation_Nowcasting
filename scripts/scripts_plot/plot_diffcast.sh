# CUDA_VISIBLE_DEVICES=0 python3 run_diffcast_plotting_consecutive.py \
#       --backbone phydnet \
#       --use_diff \
#       --dataset meteo \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Diffcast/Diffphydnet_meteo_Diffcast_on_meteonet/checkpoints/ckpt-best.pt \
#       --wandb_state 'offline' 
# CUDA_VISIBLE_DEVICES=0 python3 run_diffcast_plotting_consecutive.py \
#       --backbone phydnet \
#       --use_diff \
#       --dataset meteo \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Diffcast/Diffphydnet_meteo_Diffcast_on_meteonet/checkpoints/ckpt-best.pt \
#       --wandb_state 'offline' 


# CUDA_VISIBLE_DEVICES=0 python3 run_diffcast_plotting_consecutive.py \
#       --backbone phydnet \
#       --use_diff \
#       --dataset shanghai \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 10 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Diffcast/Diffphydnet_shanghai_Diffcast_on_shanghai/checkpoints/ckpt-best.pt \
#       --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=0 python3 run_diffcast_plotting_consecutive.py \
      --backbone phydnet \
      --use_diff \
      --dataset cikm \
      --eval \
      --plot \
      --seq_len 15 \
      --frames_in 5 \
      --frames_out 10 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Diffcast/Diffphydnet_cikm_Diffcast_on_cikm/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 