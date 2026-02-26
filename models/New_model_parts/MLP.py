import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.utilspp import RandomScheduling
from utils.wavelet_hf_loss import HF_consistency


    

class AlphaPre_Amplinet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, const_ratio, total_steps):
        super(AlphaPre_Amplinet, self).__init__()
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        self.tmlp = nn.Sequential(
            nn.Linear(pre_seq_length, aft_seq_length)
        )
        
    def forward(self, x, y, cmp_fft_loss=False):
        x_m = self.tmlp(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        return x_m

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        
        xas = self(frames_in, frames_gt, compute_loss)

        if compute_loss:
            falfcl_loss = self.falfcl(xas, frames_gt)
            loss = {'total_loss': falfcl_loss}
            return xas, loss
        else:
            return xas, None



def get_model(
    total_steps,
    const_ratio,
    T_in = 5, 
    T_out = 20,
    **kwargs
):
    model = AlphaPre_Amplinet(pre_seq_length=T_in, aft_seq_length=T_out, const_ratio=const_ratio, total_steps=total_steps)
    
    return model