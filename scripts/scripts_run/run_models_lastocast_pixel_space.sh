CUDA_VISIBLE_DEVICES=1 python3 run_alphapre_convlstm.py \
<<<<<<< HEAD
    --exp_dir sevir \
    --exp_note lastocast_on_sevir_pixel_space \
=======
    --exp_dir shanghai \
    --exp_note lastocast_on_shanghai_pixel_space \
>>>>>>> b4e31b2 (lastocast_pixel_space)
    --batch_size 4 \
    --hidden_dim 64 \
    --size_factor 1.0 \
    --backbone lastocast \
<<<<<<< HEAD
    --dataset sevir \
    --res_opt \
    --ckpt_milestone /home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/sevir/lastocast_on_sevir_pixel_space/checkpoints/ckpt-last.pt \
=======
    --dataset shanghai \
>>>>>>> b4e31b2 (lastocast_pixel_space)
    --seq_len 25 \
    --valid \
    --epochs 50 \
    --frames_in 5 \
    --frames_out 20 \
    --num_workers 8 \
    --wandb_state 'online' \
    --wandb_project_name 'Alphapre' \
<<<<<<< HEAD
    --run_name "lastocast_sevir_pixel_space" 
=======
    --run_name "lastocast_shanghai_pixel_space" 
>>>>>>> b4e31b2 (lastocast_pixel_space)
