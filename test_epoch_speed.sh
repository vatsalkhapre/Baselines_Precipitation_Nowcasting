python3 run_alphapre_convlstm_sevir_lr_latent.py --dataset meteo_lr_latent_32 --exp_dir meteo_lr_latent_32 --epochs 1 --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth" --valid

python3 run_alphapre_convlstm_sevir_lr_latent.py --dataset cikm_latent_32 --exp_dir cikm_latent_32 --epochs 1 --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth" --valid

python3 run_alphapre_convlstm_sevir_lr_latent.py --dataset shanghai_lr_latent_32 --exp_dir shanghai_lr_latent_32 --epochs 1 --ae_ckpt_path "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_SHANGHAI.pth" --valid
