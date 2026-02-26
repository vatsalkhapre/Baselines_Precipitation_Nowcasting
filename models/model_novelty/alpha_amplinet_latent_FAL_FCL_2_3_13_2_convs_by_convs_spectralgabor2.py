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
    

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        
        """
        in_channels: This will be (dim * t_in)
        out_channels: This will be (dim * t_out)
        modes1: Fourier modes to keep for Height
        modes2: Fourier modes to keep for Width
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        # Scale factor to initialize complex weights properly
        self.scale = (1 / (in_channels * out_channels))
        
        # Complex weights: [In_dim, Out_dim, Modes_H, Modes_W]
        # These weights mix Time and Channels simultaneously in the frequency domain!
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Complex multiplication using Einstein Summation
        # b: batch, i: in_channels (Time*C), o: out_channels (Time*C), h: modes_H, w: modes_W
        return torch.einsum("bihw, iohw -> bohw", input, weights)

    def forward(self, x):
        # x shape expects: [Batch, (dim * t_in), Height, Width]
        batchsize = x.shape[0]
        
        # 1. Transform spatial dimensions into Frequency Domain
        # x_ft shape: [Batch, (dim * t_in), Height, Width // 2 + 1]
        x_ft = torch.fft.rfft2(x)

        # 2. Initialize the output frequency tensor with zeros (filters out high frequencies)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # 3. Apply the Operator: Mix channels and time for the Top-Left Corner modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
            
        # 4. Apply the Operator: Mix channels and time for the Bottom-Left Corner modes
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 5. Inverse FFT back to spatial domain
        # Output shape: [Batch, (dim * t_out), Height, Width]
        x_out = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x_out

    
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
        x = x + residual
        return x
    
class AmpliNet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, weight_scale, alpha, beta, freq_multiplier, n_layers=1, mlp_ratio=2):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
        
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
            loss = 0.
          
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

class Spectral_Block(nn.Module):
    def __init__(self, dim, dim_out, modes):
        super(Block, self).__init__()
        self.spatial_temporal_interaction = nn.Sequential(
            SpectralConv2d(dim , dim_out , modes1=modes, modes2=modes),
            nn.LeakyReLU(0.5),
            # Optional: A standard 1x1 conv to mix features in pixel space after the FNO
            nn.Conv2d(dim_out,dim_out, kernel_size=1) 
        )

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