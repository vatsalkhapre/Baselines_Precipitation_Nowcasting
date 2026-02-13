import h5py
import numpy as np
import torch
from torchvision import transforms
import os
from tqdm import tqdm
from collections import OrderedDict
from models.autoencoder_kl import AutoencoderKL 

# ================= CONFIGURATION =================
INPUT_H5_PATH = '/home/vatsal/Dataserver2/Datasets/CIKM/cikm.h5'    # Input raw data
OUTPUT_H5_PATH = '/home/vatsal/Dataserver2/Datasets/Latent_32_Datasets/cikm_latent32.h5'     # Output encoded data
AE_CKPT = './Pretrained_ae_checkpoints/autoencoder_checkpoint_32_CIKM.pth'      # VAE Checkpoint
IMG_SIZE = 128
ORIGINAL_SIZE = 101                           # Original spatial size
PIXEL_SCALE = 255.0                           # Normalization constant
VAE_SCALE_FACTOR = 1.0                 # Common constant (e.g., 0.18215 for Stable Diffusion VAEs)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: Import your specific Autoencoder class here

# model = AutoencoderKL(...) 
# For now, I will assume 'model' is instantiated in the main block below.
# =================================================

# --- YOUR PROVIDED FUNCTIONS ---
def load_autoencoder_for_compression(model, checkpoint_path, device="cuda", dtype=torch.float32):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # ... (Your provided logic checks out perfectly) ...
    # Simplified for brevity in this display, but assume full logic is used:
    
    if "model" in ckpt:
        ckpt_model = ckpt["model"]
    else:
        ckpt_model = ckpt # Handle cases where ckpt is just the state_dict

    # Handle nesting
    if isinstance(ckpt_model, dict) and all(isinstance(v, dict) for v in ckpt_model.values()):
        if "autoencoder_kl" in ckpt_model:
            ckpt_state = ckpt_model["autoencoder_kl"]
            print("Hari bol")
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
    # x shape: (T, C, H, W) -> treated as (Batch, C, H, W)
    z = model.encode(x)
    
    # Check if the model returns a distribution (DiagonalGaussian) or a tensor
    if hasattr(z, 'sample'):
        return z.sample() * scale_factor
    else:
        # Some implementations return the distribution object directly, 
        # others might return a tuple or just the mean. 
        # Assuming standard LDM/Diffusers output:
        return z.sample() * scale_factor

# --- MAIN PROCESSING SCRIPT ---

def process_to_latents(model_instance):
    # 1. Setup Data Transform
    # Resize only. Normalization happens manually to match your logic.
    transform = transforms.Compose([
            transforms.CenterCrop((IMG_SIZE, IMG_SIZE)),       # padding if img_size > 101
        ])

    print(f"Opening Source: {INPUT_H5_PATH}")
    print(f"Creating Target: {OUTPUT_H5_PATH}")

    with h5py.File(INPUT_H5_PATH, 'r') as source_f, h5py.File(OUTPUT_H5_PATH, 'w') as target_f:
        
        # Copy top-level metadata (like train_len/test_len) if they exist
        for key in source_f.keys():
            if not isinstance(source_f[key], h5py.Group):
                print(f"Copying metadata key: {key}")
                target_f.create_dataset(key, data=source_f[key][()])

        # Iterate over 'train' and 'test' groups
        for split_type in ['train', 'test']:
            if split_type not in source_f:
                continue

            print(f"\nProcessing Group: {split_type}...")
            
            # Create the group in the new file
            target_grp = target_f.create_group(split_type)

            # Copy group-level attributes/metadata (like 'all_len')
            for key, val in source_f[split_type].attrs.items():
                target_grp.attrs[key] = val
            
            # Handle 'all_len' if it's a dataset inside the group
            # if 'all_len' in source_f[split_type]:
            #     target_grp.create_dataset('all_len', data=source_f[split_type]['all_len'][()])

            # Get list of valid keys (numeric strings)
            keys = [k for k in source_f[split_type].keys()]
            
            # Use tqdm for progress bar
            for key in tqdm(keys):
                # --- 1. Read Data ---
                # Shape: (T, H, W) e.g., (25, 565, 784)
                img_data = source_f[split_type][key][()]

                # --- 2. Preprocessing Pipeline ---
                # Convert to Torch -> Float
                frames = torch.from_numpy(img_data).float().to(DEVICE)
                
                # Normalize
                frames = frames / PIXEL_SCALE
                
                # Reshape for Resize: needs (Batch/Sequence, Channels, H, W) or (C, H, W)
                # Currently (T, H, W). We need (T, 1, H, W)
                frames = frames.unsqueeze(1) # Shape: (25, 1, 565, 784)
                
                # Resize
                # transforms.Resize handles (..., H, W), so (25, 1, 128, 128)
                frames = transform(frames)

                # Ensure T, C, H, W format (which it is now: 25, 1, 128, 128)
                
                # --- 3. Encode ---
                # We pass the whole sequence as a batch to the encoder
                # Output shape will be (25, Latent_C, Latent_H, Latent_W)
                latents = encode_stage(model_instance, frames, VAE_SCALE_FACTOR)

                # --- 4. Save to H5 ---
                # Move back to CPU numpy
                latents_np = latents.cpu().numpy()
                
                # Save into the corresponding group with the same key
                target_grp.create_dataset(key, data=latents_np)

    print("\n✅ Compression Complete!")


if __name__ == '__main__':
    model = AutoencoderKL(in_channels=1 , out_channels=1, down_block_types = ('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'), up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'), block_out_channels=(128, 256, 512), layers_per_block=2, latent_channels=4, norm_num_groups=32)
    model = load_autoencoder_for_compression(
        model,
        checkpoint_path = AE_CKPT,
        device="cuda",
        dtype=torch.float32
    )
    
    
    # Run processing
    process_to_latents(model)