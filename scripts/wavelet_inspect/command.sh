python3 Wavelet_exp/freq_swap_exp.py \
    --dataset_dir /home/vatsal/Dataserver2/Datasets/sevir/ \
    --code_dir . \
    --n_events 15 --wavelet db6 --level 4 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Wavelet_exp/Plots/Level4/



python3 Exp_paper/Wavelet_exp/freq_swap_exp.py --dataset cikm \
    --dataset_dir /home/vatsal/NWM/Dataset/CIKM/cikm.h5 --code_dir . \
    --img_size 128 --n_events 15 --wavelet db6 --level 3 \
    --out_dir /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Wavelet_exp/Plots/cikm/Level3/