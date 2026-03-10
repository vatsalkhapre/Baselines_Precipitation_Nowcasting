CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone phydnet \
      --dataset sevir \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 40 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Phydnet/phydnet_on_sevir/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 


CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone phydnet \
      --dataset meteo \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Phydnet/phydnet_on_meteonet/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone phydnet \
      --dataset shanghai \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 10 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Phydnet/phydnet_on_shanghai/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=2 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone phydnet \
      --dataset cikm \
      --eval \
      --plot \
      --seq_len 15 \
      --frames_in 5 \
      --frames_out 10 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Phydnet/phydnet_on_cikm/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 