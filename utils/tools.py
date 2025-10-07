import logging
from copy import deepcopy
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import torch



def exists(x):
    return x is not None       
        
def cycle(dl):
    while True:
        for data in dl:
            yield data

def print_log(message, is_main_process=True):
    if is_main_process:
        print(message)
        logging.info(message)

def show_img_info(imgs):
    print("="*30 + "Image Info" + "="*30)
    print("Tensor Shape: ",imgs.shape)
    print("Tensor DType: ",imgs.dtype)
    print("Max Value:", imgs.max())
    print("Min Value: ",imgs.min())
    print("Mean: ", imgs.mean())
        
def plot_vil(maxi):
    VIL_COLORS = [[0, 0, 0],
              [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
              [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
              [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
              [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
              [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
              [0.9607843137254902, 0.9607843137254902, 0.0],
              [0.9294117647058824, 0.6745098039215687, 0.0],
              [0.9411764705882353, 0.43137254901960786, 0.0],
              [0.6274509803921569, 0.0, 0.0],
              [0.9058823529411765, 0.0, 1.0]]

 

    VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
    VIL_LEVELS = np.linspace(0, 2000, len(VIL_LEVELS))

    cols = deepcopy(VIL_COLORS)
    lev = deepcopy(VIL_LEVELS)

    nil = cols.pop(0)
    under = cols[0]
    # over = cols.pop()
    over = cols[-1]
    cmap = ListedColormap(cols)
    cmap.set_bad(nil)
    cmap.set_under(under)
    cmap.set_over(over)
    norm = BoundaryNorm(lev, cmap.N)
    return cmap , norm

def vil_normalize(x, mini=0, maxi=76377.62):
    
    x = (x-mini)/(maxi-mini)
    return x

def convert_vil_back(x):
    x = x*11.243458
    y = np.exp(x) - 1
    return y

def unnormalize(x, min, max):
    return (x*(max - min) + min)

    
