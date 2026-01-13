import os
import h5py
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from collections import OrderedDict
from models.autoencoder_kl import AutoencoderKL
# ---------------- CONFIG ---------------- #

SRC_ROOT = "/home/vatsal/Dataserver2/Datasets/sevir/data/vil/"          # contains 2017/2018/2019
DST_ROOT = "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/sevir_lr_latent_32_resize_normalize/"         # new repo
AE_CKPT  = "./Pretrained_ae_checkpoints/autoencoder_checkpoint_32.pth"

DEVICE = "cuda:0"
LATENT_C = 4
LATENT_HW = 32
SCALE_FACTOR = 1.0             

# ---------------------------------------- #

resize = transforms.Resize(
    (128, 128),
    interpolation=transforms.InterpolationMode.BILINEAR,
)

def load_autoencoder_for_compression(
    model,
    checkpoint_path,
    device="cuda",
    dtype=torch.float32
):
    """
    model: instantiated autoencoder model (same architecture as training)
    checkpoint_path: path to .pt / .pth checkpoint
    """

    # ---- load checkpoint to CPU first (safe) ----
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    assert "model" in ckpt, "Checkpoint does not contain 'model' key"

    ckpt_model = ckpt["model"]
    
    # ---- find matching submodel key ----
    model_keys = list(model.state_dict().keys())
    
    ckpt_keys = list(ckpt_model.keys())
    
    # If checkpoint saved multiple submodels, pick autoencoder
    if isinstance(ckpt_model, dict) and all(isinstance(v, dict) for v in ckpt_model.values()):
        # typical structure: ckpt['model']['autoencoder_kl']
        if len(ckpt_model) == 1:
            ckpt_state = list(ckpt_model.values())[0]
        else:
            # explicitly choose autoencoder
            
            ckpt_state = ckpt_model.get("autoencoder_kl", None)
            if ckpt_state is None:
                raise KeyError("autoencoder_kl not found in checkpoint")
            else:
                print("Hari bol")
                
    else:
        ckpt_state = ckpt_model

    # ---- strip 'module.' if present ----
    new_state_dict = OrderedDict()
    for k, v in ckpt_state.items():
        if k.startswith("module."):
            k = k[7:]
        elif k.startswith("net."):
            k = k[4:]
        new_state_dict[k] = v

    
    # ---- load weights ----
    model.load_state_dict(new_state_dict, strict=True)

    # ---- move to device and eval ----
    model.to(device=device, dtype=dtype)
    model.eval()

    # ---- freeze params (important for compression) ----
    for p in model.parameters():
        p.requires_grad = False

    print("✅ Autoencoder loaded for compression")
    return model


@torch.no_grad()
def encode_stage(model, x, scale_factor):
    z = model.encode(x)
    return z.sample() * scale_factor


@torch.no_grad()
def decode_stage(model, z, scale_factor):
    z = z / scale_factor
    return model.decode(z)

# ---------------- PROCESS ONE FILE ---------------- #

def process_h5(src_path, dst_path, autoencoder):
    scale_factor = 1.0
    print("src_path", src_path)
    with h5py.File(src_path, "r") as f:
        vil = f["vil"]            # (N, 384, 384, 49)
        N, H, W, T = vil.shape

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        with h5py.File(dst_path, "w") as fout:
            dset = fout.create_dataset(
                "vil_latent",
                shape=(N, T, LATENT_C, LATENT_HW, LATENT_HW),
                dtype=np.float16,
                compression="gzip",
                compression_opts=4,
            )

            for i in tqdm(range(N), desc=os.path.basename(src_path)):
                # (384, 384, 49) -> (49, 384, 384)
                x = vil[i].transpose(2, 0, 1)

                x = torch.from_numpy(x).float().to(DEVICE)   # uint8 → float
                # x = x / 255.0                                # normalize FIRST
                # x = resize(x)    
                x = resize(x) 
                x = x / 255.0
                x = x.unsqueeze(1)                            # (T,1,128,128)
                x = x.unsqueeze(0)                            # (1,T,1,128,128)

                B, T, C, H, W = x.shape

                x_flat = x.view(B * T, C, H, W).contiguous()

                z_flat = encode_stage(
                    autoencoder,
                    x_flat,
                    scale_factor=scale_factor
                )
                
                # z_decoded = decode_stage(autoencoder, z_flat, scale_factor)

                # mse = torch.mean(z_decoded-x_flat)**2

                # psnr = -10 * torch.log10(mse)
                # print(psnr.item())
                # z_flat: [B*T, 4, 32, 32]
                z = z_flat.view(B, T, z_flat.shape[1], z_flat.shape[2], z_flat.shape[3])

                dset[i] = z.squeeze(0).cpu().half().numpy()

# ---------------- MAIN LOOP ---------------- #

def main():
    model = AutoencoderKL(in_channels=1 , out_channels=1, down_block_types = ('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'), up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'), block_out_channels=(128, 256, 512), layers_per_block=2, latent_channels=4, norm_num_groups=32)
    model = load_autoencoder_for_compression(
        model,
        checkpoint_path = AE_CKPT,
        device="cuda",
        dtype=torch.float32
    )
     
    for year in ["2017", "2018", "2019"]:
        src_year = os.path.join(SRC_ROOT, year)
        dst_year = os.path.join(DST_ROOT, year)

        for fname in sorted(os.listdir(src_year)):
            if not fname.endswith(".h5"):
                continue

            src_file = os.path.join(src_year, fname)
            dst_file = os.path.join(dst_year, fname)

            if os.path.exists(dst_file):
                print("exists!")
                continue
            print(f"Processing {src_file}")
            process_h5(src_file, dst_file, model)

    print("✅ All files processed successfully")

if __name__ == "__main__":
    main()

   