# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
#       --backbone alphapre \
#       --dataset sevir \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 40 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/alphapre/alphapre_on_sevir/checkpoints/ckpt-best.pt \
#       --wandb_state 'offline' 


# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
#       --backbone alphapre \
#       --dataset meteo \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Alphapre_other_datasets/alphapre_meteo_Training_100epochs_final_resume/checkpoints/ckpt-best.pt \
#       --wandb_state 'offline' 

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
#       --backbone alphapre \
#       --dataset shanghai \
#       --eval \
#       --plot \
#       --seq_len 25 \
#       --frames_in 5 \
#       --frames_out 20 \
#       --num_workers 8 \
#       --plot_stride 10 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Alphapre_other_datasets/Alphapre_shanghai_Training_150epochs/checkpoints/ckpt-best.pt \
      # --wandb_state 'offline' 

# CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
#       --backbone alphapre \
#       --dataset cikm \
#       --eval \
#       --plot \
#       --seq_len 15 \
#       --frames_in 5 \
#       --frames_out 10 \
#       --num_workers 8 \
#       --plot_stride 20 \
#       --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Baselines/Alphapre_other_datasets/Alphapre_cikm/checkpoints/ckpt-best.pt \
#       --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset meteo \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_meteonet/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset shanghai \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 10 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_shanghai/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset cikm \
      --eval \
      --plot \
      --seq_len 15 \
      --frames_in 5 \
      --frames_out 10 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_cikm/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset meteo \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_meteonet/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset shanghai \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 10 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_shanghai/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset cikm \
      --eval \
      --plot \
      --seq_len 15 \
      --frames_in 5 \
      --frames_out 10 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_cikm/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset meteo \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_meteonet/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset shanghai \
      --eval \
      --plot \
      --seq_len 25 \
      --frames_in 5 \
      --frames_out 20 \
      --num_workers 8 \
      --plot_stride 10 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_shanghai/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 

CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm_with_plotting_consecutive.py \
      --backbone earthformer \
      --dataset cikm \
      --eval \
      --plot \
      --seq_len 15 \
      --frames_in 5 \
      --frames_out 10 \
      --num_workers 8 \
      --plot_stride 20 \
      --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Qualitative_analysis/Earthformer/earthformer_on_cikm/checkpoints/ckpt-best.pt \
      --wandb_state 'offline' 