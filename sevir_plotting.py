import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import os

# ---- CONFIG ---- #
H5_PATH = "/home/vatsal/Dataserver2/Datasets/sevir/data/vil/2018/SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5"   # <-- change this
FRAME_IDX = 0      # which timestep out of 49
OUT_DIR = "./vil_frames"

# ---- SEVIR VIL colormap ---- #
COLOR_MAP = [
    [0, 0, 0],
    [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
    [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
    [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
    [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
    [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
    [0.9607843137254902, 0.9607843137254902, 0.0],
    [0.9294117647058824, 0.6745098039215687, 0.0],
    [0.9411764705882353, 0.43137254901960786, 0.0],
    [0.6274509803921569, 0.0, 0.0],
    [0.9058823529411765, 0.0, 1.0],
]

PIXEL_SCALE = 255.0
BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]

cmap = colors.ListedColormap(COLOR_MAP)
norm = colors.BoundaryNorm(BOUNDS, cmap.N)

os.makedirs(OUT_DIR, exist_ok=True)

# ---- READ & SAVE ---- #
with h5py.File(H5_PATH, "r") as f:
    vil = f["vil"]  # shape: (N, 384, 384, 49)
    N = vil.shape[0]
    print(f"Dataset shape: {vil.shape} — saving frame {FRAME_IDX} from all {N} events")

    for i in range(N):
        frame = vil[i, :, :, FRAME_IDX].astype(np.float64)  # (384, 384)

        # Apply colormap manually to get an RGBA image, then drop alpha
        rgba = cmap(norm(frame))  # (384, 384, 4) float in [0,1]
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        # Save as a clean square image — no axes, borders, or whitespace
        fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
        ax.imshow(rgb, interpolation="nearest")
        ax.axis("off")
        ax.set_aspect("equal")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            os.path.join(OUT_DIR, f"event_{i:04d}.png"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

        if (i + 1) % 50 == 0:
            print(f"  saved {i + 1}/{N}")

print(f"Done — {N} images saved to {OUT_DIR}/")





