import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.utilspp import RandomScheduling
from utils.wavelet_hf_loss import HF_consistency


    

class AlphaPre_Amplinet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, const_ratio, total_steps):
        super(AlphaPre_Amplinet, self).__init__()
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length

        self.tmlp = nn.Sequential(nn.Linear(pre_seq_length, aft_seq_length))

        self.convin = nn.Sequential(ResnetBlock(dim, hidden_dim),
                                    ResnetBlock(hidden_dim, hidden_dim),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))

        self.convout = nn.Sequential(ResnetBlock(hidden_dim, hidden_dim),
                                     ResnetBlock(hidden_dim, hidden_dim),
                                     nn.Conv2d(hidden_dim, dim, kernel_size=1))
        
        
    def forward(self, x, y, cmp_fft_loss=False):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)
        x_m = self.tmlp(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        x = rearrange(x_m, 'b t c h w -> (b t) c h w',  t=self.aft_seq_length)
        x = self.convout(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)
        return x

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        
        xas = self(frames_in, frames_gt, compute_loss)
        loss = 0.
        if compute_loss:
            falfcl_loss = self.falfcl(xas, frames_gt)
            loss = {'total_loss': falfcl_loss}
            return xas, loss
        else:
            return xas, None


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', groupnorm=True):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding = kernel_size//2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros'): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


def get_model(
    total_steps,
    const_ratio,
    T_in = 5, 
    T_out = 20,
    img_channels=1,
    dim = 64,
    **kwargs
):
    model = AlphaPre_Amplinet(pre_seq_length=T_in, aft_seq_length=T_out, dim=img_channels, hidden_dim=dim,  const_ratio=const_ratio, total_steps=total_steps)
    
    return model