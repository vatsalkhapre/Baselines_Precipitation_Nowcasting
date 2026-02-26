import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from utils.utilspp import RandomScheduling
from utils.wavelet_hf_loss import HF_consistency

class GaborLayer(nn.Module):
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0, freq_multiplier = 1.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.linear.weight.data *= weight_scale * torch.sqrt(self.gamma[:, None])
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        self.param = nn.Parameter(torch.rand(out_features))
        self.freq_multiplier = freq_multiplier
        
        return
 
    def forward(self, x):
        D = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu.T
        )
        return torch.sin(self.freq_multiplier*self.param*self.linear(x)) * torch.exp(-0.5 * D * self.gamma[None, :])
    

class AmpTimeCell(nn.Module):
    def __init__(self, t_in, t_out, size_factor=1):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out*size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out*size_factor), t_out),
        )
        self.scale = 0.02

        self.w1 = nn.Parameter((self.scale * torch.randn(2, t_in, t_out*size_factor)))
        self.b1 = nn.Parameter((self.scale * torch.randn(2, 1, 1, 1, t_out*size_factor)))
        self.w2 = nn.Parameter((self.scale * torch.randn(2, t_out*size_factor, t_out)))
        self.b2 = nn.Parameter((self.scale * torch.randn(2, 1, 1, 1, t_out)))
    
    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        bias = self.tmlp(x)
        xf = torch.fft.rfft2(x, dim=[2,3], norm="ortho")
        x1_real = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[0]) - \
                  torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[1]) + \
                  self.b1[0]
        x1_imag = torch.einsum('bchwt,to->bchwo', xf.real, self.w1[1]) + \
                  torch.einsum('bchwt,to->bchwo', xf.imag, self.w1[0]) + \
                  self.b1[1]
        x1_real, x1_imag = F.relu(x1_real), F.relu(x1_imag)
        
        x2_real = torch.einsum('bchwt,to->bchwo', x1_real, self.w2[0]) - \
                  torch.einsum('bchwt,to->bchwo', x1_imag, self.w2[1]) + \
                  self.b2[0]
        x2_imag = torch.einsum('bchwt,to->bchwo', x1_real, self.w2[1]) + \
                  torch.einsum('bchwt,to->bchwo', x1_imag, self.w2[0]) + \
                  self.b2[1]

        x2 = torch.view_as_complex(torch.stack([x2_real, x2_imag], dim=-1))
        x = torch.fft.irfft2(x2, dim=[2,3], norm="ortho")
        x = x + bias
        return x.permute(0,4,1,2,3)
    
class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, weight_scale, alpha, beta, freq_multiplier, size_factor=1.0,
        ):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.gabor = GaborLayer(t_in, t_out, weight_scale, alpha, beta, freq_multiplier)
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out*size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out*size_factor), t_out),
        )
        # self.amptime =  AmpTimeCell(t_in, t_out)
        self.fusion = nn.Conv3d(2*dim, dim, kernel_size=1)
        self.conv = nn.Sequential(ResnetBlock(dim*t_out, dim*t_out),
                                     ResnetBlock(dim*t_out, dim*t_out),
                                     nn.Conv2d(dim*t_out, dim*t_out, kernel_size=3, padding=1))
    def forward(self, x):
        residual = self.gabor(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        residual2 = self.tmlp(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        out = torch.cat([residual, residual2], dim=2)
        out = out.permute(0,2,1,3,4)  
        x= self.fusion(out)
        x = x.permute(0,2,1,3,4) 
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        return x
    
class AmpliNet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, weight_scale, alpha, beta, freq_multiplier, n_layers=1, mlp_ratio=2):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
        # self.tmlp = nn.Sequential(
        #     nn.Linear(pre_seq_length, int(aft_seq_length*mlp_ratio)),
        #     nn.SELU(True),
        #     nn.Linear(int(aft_seq_length*mlp_ratio), aft_seq_length),
        # )
        self.convin = nn.Sequential(ResnetBlock(dim, hidden_dim),
                                    ResnetBlock(hidden_dim, hidden_dim),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.amplist = nn.ModuleList([
            AmpCell(pre_seq_length if i==0 else aft_seq_length, aft_seq_length, hidden_dim, weight_scale, alpha, beta, freq_multiplier) for i in range(n_layers)
        ])
        self.convout = nn.Sequential(ResnetBlock(hidden_dim, hidden_dim),
                                     ResnetBlock(hidden_dim, hidden_dim),
                                     nn.Conv2d(hidden_dim, dim, kernel_size=1))

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)
        # x_ = x.permute(0,2,3,4,1)
        # xr = self.tmlp(x_)
        # xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for ampcell in self.amplist:
            x = ampcell(x)
        # x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convout(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return x
    
class AlphaPre_Amplinet(nn.Module):
    def __init__(self, weight_scale, alpha, beta, freq_multiplier, total_steps,const_ratio, pre_seq_length, aft_seq_length, input_shape, input_dim, 
                 hidden_dim, n_layers, spec_num=20, kernel_size=1, bias=1, 
                 pha_weight=0.01, anet_weight=0.1, amp_weight=0.01, aweight_stop_steps=10000):
        super(AlphaPre_Amplinet, self).__init__()
        self.amplinet = AmpliNet(pre_seq_length, aft_seq_length, input_dim, hidden_dim, weight_scale, alpha, beta, freq_multiplier)
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



def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )

def get_model(
    weight_scale,
    alpha,
    beta,
    freq_multiplier,
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
    model = AlphaPre_Amplinet(weight_scale, alpha, beta, freq_multiplier, total_steps,const_ratio, pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape, input_dim=img_channels, 
                     hidden_dim=dim, n_layers=n_layers, spec_num=spec_num,
                     pha_weight=pha_weight, anet_weight=anet_weight, amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
                     )
    
    return model