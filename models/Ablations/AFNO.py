import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.utilspp import RandomScheduling
from utils.wavelet_hf_loss import HF_consistency

class AFNO3DCore(nn.Module):

    def __init__(self, hidden_size, num_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1.0,
                 hidden_size_factor=1):

        super().__init__()
        assert hidden_size % num_blocks == 0

        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(
                2, num_blocks, self.block_size,
                self.block_size * hidden_size_factor
            )
        )
        self.b1 = nn.Parameter(
            self.scale * torch.randn(
                2, num_blocks,
                self.block_size * hidden_size_factor
            )
        )

        self.w2 = nn.Parameter(
            self.scale * torch.randn(
                2, num_blocks,
                self.block_size * hidden_size_factor,
                self.block_size
            )
        )
        self.b2 = nn.Parameter(
            self.scale * torch.randn(
                2, num_blocks, self.block_size
            )
        )

    def forward(self, x):
        # x: (B, hidden, T, H, W)

        B, C, T, H, W = x.shape
        bias = x
        x = x.float()

        x = torch.fft.rfftn(x, dim=(2,3,4), norm="ortho")

        x = x.reshape(B, C, T, H, W//2 + 1)
        x = x.view(B, self.num_blocks,
                   self.block_size,
                   T, H, W//2 + 1)
        x = x.permute(0, 3, 4, 5, 1, 2)

        Kt = int(T * self.hard_thresholding_fraction)
        Kh = int(H * self.hard_thresholding_fraction)
        Kw = int((W//2 + 1) * self.hard_thresholding_fraction)

        xr = x[:, :Kt, :Kh, :Kw].real
        xi = x[:, :Kt, :Kh, :Kw].imag

        o1r = torch.einsum('...bi,bio->...bo', xr, self.w1[0]) \
              - torch.einsum('...bi,bio->...bo', xi, self.w1[1]) \
              + self.b1[0]

        o1i = torch.einsum('...bi,bio->...bo', xi, self.w1[0]) \
              + torch.einsum('...bi,bio->...bo', xr, self.w1[1]) \
              + self.b1[1]

        o2r = torch.einsum('...bi,bio->...bo', o1r, self.w2[0]) \
              - torch.einsum('...bi,bio->...bo', o1i, self.w2[1]) \
              + self.b2[0]

        o2i = torch.einsum('...bi,bio->...bo', o1i, self.w2[0]) \
              + torch.einsum('...bi,bio->...bo', o1r, self.w2[1]) \
              + self.b2[1]

        x[:, :Kt, :Kh, :Kw] = torch.complex(o2r, o2i)

        x = x.permute(0, 4, 5, 1, 2, 3).reshape(B, C, T, H, W//2 + 1)

        x = torch.fft.irfftn(x, s=(T, H, W),
                             dim=(2,3,4),
                             norm="ortho")

        return x + bias
    
class AFNO3DForecast(nn.Module):

    def __init__(self, T_in, T_out, C,
                 hidden_size=64,
                 num_blocks=8):

        super().__init__()

        self.T_out = T_out
        self.hidden_size = hidden_size

        # Lifting
        self.lift = nn.Conv3d(C, hidden_size, kernel_size=1)

        # Spectral core
        self.afno = AFNO3DCore(hidden_size, num_blocks)

        # Time projection
        self.time_proj = nn.Conv3d(
            hidden_size,
            hidden_size * T_out,
            kernel_size=(T_in,1,1)
        )

        # Output projection
        self.proj = nn.Conv3d(hidden_size, C, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W)

        B, T, C, H, W = x.shape

        x = x.permute(0,2,1,3,4)  # (B,C,T,H,W)

        x = self.lift(x)          # (B,hidden,T,H,W)

        x = self.afno(x)          # same shape

        x = self.time_proj(x)     # (B,hidden*T_out,1,H,W)

        x = x.squeeze(2)          # (B,hidden*T_out,H,W)

        x = x.view(B, self.T_out,
                   self.hidden_size, H, W)

        x = x.permute(0,2,1,3,4)   # (B,hidden,T_out,H,W)

        x = self.proj(x)          # (B,C,T_out,H,W)

        x = x.permute(0,2,1,3,4)   # (B,T_out,C,H,W)

        return x
    
class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.afno = AFNO3DForecast(t_in, t_out, 4)
        

    def forward(self, x):
        out = self.afno(x)
        return out
    
class AmpliNet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=1):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
    
        
        self.amplist = nn.ModuleList([
            AmpCell(pre_seq_length if i==0 else aft_seq_length, aft_seq_length, hidden_dim) for i in range(n_layers)
        ])
        
    def forward(self, x):
    
        # x_ = x.permute(0,2,3,4,1)
        # xr = self.tmlp(x_)
        # xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for ampcell in self.amplist:
            x = ampcell(x)
        # x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
    
        return x
    
class AlphaPre_Amplinet(nn.Module):
    def __init__(self, total_steps,const_ratio, pre_seq_length, aft_seq_length, input_shape, input_dim, 
                 hidden_dim, n_layers, spec_num=20, kernel_size=1, bias=1, 
                 pha_weight=0.01, anet_weight=0.1, amp_weight=0.01, aweight_stop_steps=10000):
        super(AlphaPre_Amplinet, self).__init__()
        self.amplinet = AmpliNet(pre_seq_length, aft_seq_length, input_dim, hidden_dim)
        self.input_shape, self.input_dim = input_shape, input_dim
        self.hidden_dim = hidden_dim
        self.spec_num = spec_num
        self.pha_weight = pha_weight
        self.anet_weight = anet_weight
        self.amp_weight = amp_weight
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.falfcl = RandomScheduling(total_steps, 1, const_ratio)
        # self.hfloss = HF_consistency()
        self.itr = 0
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate =  self.amp_weight/self.aweight_stop_steps

        h, w = input_shape
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[...,:spec_num,:spec_num] = 1.
        spec_mask[...,-spec_num:,:spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)
        
    def forward(self, x, y, cmp_fft_loss=False): # x:[b,t,c,h,w]
        self.itr += 1
        xas = self.amplinet(x)
        # xas = torch.sigmoid(xas)
        return xas

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            if self.itr < self.aweight_stop_steps:
                self.amp_weight -= self.sampling_changing_rate
            else:
                self.amp_weight  = 0.

            loss = 0.
            
            # frames_fft = torch.fft.rfft2(frames_gt)
            # frames_abs = torch.abs(frames_fft)
            # xas_fft = torch.fft.rfft2(xas)
            # xas_abs = torch.abs(xas_fft)
            # amp_loss = self.criterion(xas_abs, frames_abs)
            # loss += self.amp_weight*amp_loss
            falfcl_loss = self.falfcl(xas, frames_gt)
            # hfloss = self.hfloss(xas, frames_gt)
            # total_loss = falfcl_loss   #Place correct weights here
            loss = {'total_loss': falfcl_loss}
            return xas, loss
        else:
            return xas, None



def get_model(
    total_steps,
    const_ratio,
    img_channels=1,
    dim = 64,
    T_in = 5, 
    T_out = 20,
    input_shape = (128,128),
    n_layers = 3,
    spec_num = 20,
    pha_weight=0.01, 
    anet_weight=0.1,
    amp_weight=0.01,
    aweight_stop_steps=10000,
    **kwargs
):
    model = AlphaPre_Amplinet(total_steps,const_ratio, pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape, input_dim=img_channels, 
                     hidden_dim=dim, n_layers=n_layers, spec_num=spec_num,
                     pha_weight=pha_weight, anet_weight=anet_weight, amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
                     )
    
    return model