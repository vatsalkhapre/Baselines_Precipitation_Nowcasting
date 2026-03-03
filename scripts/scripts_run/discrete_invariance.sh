# for f in 1.5
# do
#     for a in 1.0
#     do 
#         for b in 1.0
#         do
#             for weight_scale in 1.0
#             do
#                 for i in 384
#                 do
#                 CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_discrete_invariance.py \
#                     --backbone our_model \
#                     --dataset shanghai \
#                     --img_size ${i} \
#                     --exp_dir Trying_dicrete_invariance \
#                     --exp_note "our_model_shanghai_${weight_scale}_${a}_${b}_${f}_${i}" \
#                     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" \
#                     --eval \
#                     --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/shanghai_latent/amplinet_latent_falfcl_only_2_3_13_2_gabor2_shanghai_lr_latent_32_amplinet_latent_falfcl_only_2_3_13_2_gabor2_1.0_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
#                     --seq_len 25 \
#                     --frames_in 5 \
#                     --img_channel 4 \
#                     --frames_out 20 \
#                     --weight_scale ${weight_scale} \
#                     --alpha ${a} \
#                     --beta ${b} \
#                     --freq_multiplier ${f} \
#                     --num_workers 8 \
#                     --wandb_state 'offline' 
#                 done
#             done
#         done
#     done
# done


for f in 1.5
do
    for a in 1.0
    do 
        for b in 1.0
        do
            for weight_scale in 1.5
            do
                for i in 64 256 384
                do
                CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_discrete_invariance.py \
                    --backbone our_model \
                    --dataset meteo \
                    --img_size ${i} \
                    --exp_dir Trying_dicrete_invariance \
                    --exp_note "our_model_meteo_${weight_scale}_${a}_${b}_${f}_${i}" \
                    --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" \
                    --eval \
                    --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/meteonet_latent/w_o_AFNO/First_best/amplinet_latent_falfcl_only_2_3_13_2_gabor2_meteo_lr_latent_32_amplinet_latent_falfcl_only_2_3_13_2_gabor2_1.5_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
                    --seq_len 25 \
                    --frames_in 5 \
                    --img_channel 4 \
                    --frames_out 20 \
                    --weight_scale ${weight_scale} \
                    --alpha ${a} \
                    --beta ${b} \
                    --freq_multiplier ${f} \
                    --num_workers 8 \
                    --wandb_state 'offline' 
                done
            done
        done
    done
done


# for f in 1.5
# do
#     for a in 1.0
#     do 
#         for b in 1.0
#         do
#             for weight_scale in 1.0
#             do
#                 for i in 64 256
#                 do
#                 CUDA_VISIBLE_DEVICES=0 python3 run_alphapre_convlstm_discrete_invariance.py \
#                     --backbone our_model \
#                     --dataset cikm \
#                     --img_size ${i} \
#                     --exp_dir Trying_dicrete_invariance \
#                     --exp_note "our_model_cikm_${weight_scale}_${a}_${b}_${f}_${i}" \
#                     --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" \
#                     --eval \
#                     --ckpt_milestone /home/vatsal/Dataserver2/ECCV26/Best_models/cikm_latent/w_o_AFNO/amplinet_latent_falfcl_only_2_3_13_2_gabor2_cikm_latent_32_amplinet_latent_falfcl_only_2_3_13_2_gabor2_1.0_1.0_1.0_1.5/checkpoints/ckpt-best.pt \
#                     --seq_len 15 \
#                     --frames_in 5 \
#                     --img_channel 4 \
#                     --frames_out 10 \
#                     --weight_scale ${weight_scale} \
#                     --alpha ${a} \
#                     --beta ${b} \
#                     --freq_multiplier ${f} \
#                     --num_workers 8 \
#                     --wandb_state 'offline' 
#                 done
#             done
#         done
#     done
# done