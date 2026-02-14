import h5py
import numpy as np
import torch
from torchvision import transforms
import os
from tqdm import tqdm
from collections import OrderedDict
from models.autoencoder_kl import AutoencoderKL 

# ================= CONFIGURATION =================
INPUT_H5_PATH = '/home/vatsal/Dataserver2/Datasets/Meteonet/meteo_radar.h5'           # Input raw data
OUTPUT_H5_PATH = '/home/vatsal/Dataserver2/Datasets/Latent_32_Datasets/meteonet_latent_32/meteonet_latent32.h5'  # Output encoded data
AE_CKPT = './Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth'  # VAE Checkpoint
IMG_SIZE = 128
ORIGINAL_SHAPE = (565, 784)                   # Original spatial size (H, W)
PIXEL_SCALE = 90.0                            # Normalization constant
VAE_SCALE_FACTOR = 1.0                        # Scale factor for latents
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =================================================

def load_autoencoder_for_compression(model, checkpoint_path, device="cuda", dtype=torch.float32):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    if "model" in ckpt:
        ckpt_model = ckpt["model"]
    else:
        ckpt_model = ckpt

    if isinstance(ckpt_model, dict) and all(isinstance(v, dict) for v in ckpt_model.values()):
        if "autoencoder_kl" in ckpt_model:
            ckpt_state = ckpt_model["autoencoder_kl"]
            print("Found autoencoder_kl key")
        elif len(ckpt_model) == 1:
            ckpt_state = list(ckpt_model.values())[0]
        else:
             ckpt_state = ckpt_model
    else:
        ckpt_state = ckpt_model

    new_state_dict = OrderedDict()
    for k, v in ckpt_state.items():
        if k.startswith("module."): k = k[7:]
        elif k.startswith("net."): k = k[4:]
        new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    print("✅ Autoencoder loaded for compression")
    return model


@torch.no_grad()
def encode_stage(model, x, scale_factor):
    """
    Encode input tensor to latent space.
    
    Args:
        model: Autoencoder model
        x: Input tensor of shape (T, C, H, W)
        scale_factor: Scale factor for latents
        
    Returns:
        Latent tensor
    """
    z = model.encode(x)
    
    if hasattr(z, 'sample'):
        return z.sample() * scale_factor
    else:
        return z.sample() * scale_factor


def build_transform(img_size):
    """
    Build transform with Resize.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
    ])
    return transform


def process_to_latents(model_instance):
    """
    Process h5 file and encode to latents.
    
    Input h5 structure:
        h5['train']['0'][()] -> shape (25, 565, 784)
        h5['test']['0'][()]
    
    Processing pipeline (matching preprocess_h5_resize.py):
        1. numpy (T, H, W) = (25, 565, 784)
        2. float32
        3. normalize /90.0
        4. torch
        5. Resize -> (25, 128, 128)
        6. unsqueeze -> (25, 1, 128, 128)
        7. encode -> (25, latent_C, latent_H, latent_W)
    """
    
    # Build transform (Resize)
    transform = build_transform(IMG_SIZE)
    print(f"Transform: Resize from {ORIGINAL_SHAPE[0]}x{ORIGINAL_SHAPE[1]} to {IMG_SIZE}x{IMG_SIZE}")

    print(f"Opening Source: {INPUT_H5_PATH}")
    print(f"Creating Target: {OUTPUT_H5_PATH}")

    with h5py.File(INPUT_H5_PATH, 'r') as source_f, h5py.File(OUTPUT_H5_PATH, 'w') as target_f:
        
        # Copy top-level metadata (like train_len, test_len) if they exist
        for key in source_f.keys():
            if not isinstance(source_f[key], h5py.Group):
                print(f"Copying metadata key: {key}")
                target_f.create_dataset(key, data=source_f[key][()])

        # Iterate over 'train' and 'test' groups (no 'valid' in source h5)
        for split_type in ['train', 'test']:
            if split_type not in source_f:
                continue

            print(f"\nProcessing Group: {split_type}...")
            
            # Create the group in the new file
            target_grp = target_f.create_group(split_type)

            # Copy group-level attributes/metadata
            for key, val in source_f[split_type].attrs.items():
                target_grp.attrs[key] = val
            
            # Get list of sample keys (numeric strings: '0', '1', '2', ...)
            keys = [k for k in source_f[split_type].keys()]
            # Sort numerically
            keys = sorted(keys, key=lambda x: int(x))
            
            # Use tqdm for progress bar
            for key in tqdm(keys, desc=f"Encoding {split_type}"):
                # --- 1. Read Data ---
                # Shape: (T, H, W) = (25, 565, 784)
                img_data = source_f[split_type][key][()]

                # --- 2. Preprocessing Pipeline (matching preprocess_h5_resize.py) ---
                
                # Step 1: Convert to float32 numpy
                img_data = img_data.astype(np.float32)
                
                # Step 2: Normalize by /90.0
                img_data = img_data / PIXEL_SCALE
                
                # Step 3: Convert to torch
                frames = torch.from_numpy(img_data)  # Shape: (25, 565, 784)
                
                # Step 4: Apply transform (Resize)
                frames = transform(frames)  # Shape: (25, 128, 128)
                
                # Step 5: Unsqueeze for channel dimension
                frames = frames.unsqueeze(1)  # Shape: (25, 1, 128, 128)
                
                # Move to device
                frames = frames.to(DEVICE)

                # --- 3. Encode ---
                # Pass the whole sequence as a batch to the encoder
                # Input shape: (25, 1, 128, 128)
                # Output shape: (25, latent_C, latent_H, latent_W)
                latents = encode_stage(model_instance, frames, VAE_SCALE_FACTOR)

                # --- 4. Save to H5 ---
                # Move back to CPU numpy
                latents_np = latents.cpu().numpy()
                
                # Save into the corresponding group with the same key
                target_grp.create_dataset(key, data=latents_np)

    print("\n✅ Compression Complete!")
    print(f"Output saved to: {OUTPUT_H5_PATH}")


if __name__ == '__main__':
    print("="*60)
    print("H5 Latent Compression (Resize)")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Input:          {INPUT_H5_PATH}")
    print(f"  Output:         {OUTPUT_H5_PATH}")
    print(f"  Checkpoint:     {AE_CKPT}")
    print(f"  Original size:  {ORIGINAL_SHAPE[0]}x{ORIGINAL_SHAPE[1]}")
    print(f"  Target size:    {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Pixel scale:    {PIXEL_SCALE}")
    print(f"  VAE scale:      {VAE_SCALE_FACTOR}")
    print(f"  Device:         {DEVICE}")
    
    # Initialize model
    model = AutoencoderKL(
        in_channels=1,
        out_channels=1,
        down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
        up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
        block_out_channels=(128, 256, 512),
        layers_per_block=2,
        latent_channels=4,
        norm_num_groups=32
    )
    
    model = load_autoencoder_for_compression(
        model,
        checkpoint_path=AE_CKPT,
        device=DEVICE,
        dtype=torch.float32
    )
    
    # Run processing
    process_to_latents(model)
