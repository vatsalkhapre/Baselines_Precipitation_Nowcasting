import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

class FNOAmpTimeCell(nn.Module):
    def __init__(self, t_in, t_out, modes1=32, modes2=32, width=None):
        super().__init__()
        self.t_in = t_in
        self.t_out = t_out
        self.modes1 = modes1 # Modes to keep in height
        self.modes2 = modes2 # Modes to keep in width
        
        # If width is not provided, we stick to the original logic
        # But usually FNO projects channels up. Here "Channels" are Time steps.
        
        self.scale = 1 / (t_in * t_out)
        
        # Weights: [t_in, t_out, modes1, modes2]
        # Complex weights to mix Time per Frequency Mode
        self.weights1 = nn.Parameter(self.scale * torch.randn(t_in, t_out, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(t_in, t_out, self.modes1, self.modes2, dtype=torch.cfloat))
        
        # Residual path (same as original tmlp)
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out)),
            nn.SELU(True),
            nn.Linear(int(t_out), t_out),
        )

    # Complex multiplication: 
    # (Batch, Channel, ModesH, ModesW, Tin) * (Tin, Tout, ModesH, ModesW) -> (Batch, Channel, ModesH, ModesW, Tout)
    def compl_mul2d(self, input, weights):
        # input: (B, C, H, W, Tin) -> permute to (B, C, Tin, H, W) for easier handling?
        # Let's keep your dimension ordering: (B, C, H, W, T)
        
        # We need to contract T_in and keep B, C, H, W, T_out
        # Einstein Summation:
        # b c h w i : Batch, Channel, Height(mode), Width(mode), Time_In
        # i o h w   : Time_In, Time_Out, Height(mode), Width(mode)
        # -> b c h w o
        return torch.einsum("bchwi, iohw -> bchwo", input, weights)

    def forward(self, x):
        # x shape: [B, T_in, C, H, W]
        B, T_in, C, H, W = x.shape
        
        # 1. Residual (Bias) Path - same as original
        # Permute to [B, C, H, W, T]
        x_res = x.permute(0, 2, 3, 4, 1) 
        bias = self.tmlp(x_res)

        # 2. Spectral Path
        # FFT
        x_ft = torch.fft.rfft2(x_res, dim=(2, 3), norm="ortho")
        
        # Initialize output in frequency domain (zeros)
        # x_ft shape: [B, C, H, W//2 + 1, T]
        out_ft = torch.zeros(B, C, H, W // 2 + 1, self.t_out, device=x.device, dtype=torch.cfloat)
        
        # 3. FNO Operations (Apply weights only to lower corners/modes)
        # Corner 1: Top-Left
        out_ft[:, :, :self.modes1, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2, :], self.weights1)
    

        # Corner 2: Bottom-Left (handling periodicity in frequency)
        out_ft[:, :, -self.modes1:, :self.modes2, :] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2, :], self.weights2)

        out_ft_real = F.relu(out_ft.real)
        out_ft_imaginary = F.relu(out_ft.imag)
        out_ft = torch.view_as_complex(torch.stack([out_ft_real, out_ft_imaginary], dim=-1))

        # 4. Inverse FFT
        x = torch.fft.irfft2(out_ft, s=(H, W), dim=(2, 3), norm="ortho")
        
        # 5. Combine + Activation
        # Note: Original AmpTimeCell used SplitReLU inside spectral domain. 
        # FNO usually applies activation in Spatial Domain.
        # We will follow standard FNO practice here: IFFT -> Add Bias -> Gelu/Relu
        
        x = x + bias
        # x = F.silu(x) # Using SiLU (Swish) as is common in modern FNOs
        
        # Permute back to [B, T_out, C, H, W]
        return x.permute(0, 4, 1, 2, 3)
    
class AmpCell(nn.Module):
    def __init__(self, t_in, t_out, dim, size_factor=1.0):
        super().__init__()
        self.t_in, self.t_out = t_in, t_out
        self.tmlp = nn.Sequential(
            nn.Linear(t_in, int(t_out*size_factor)),
            nn.SELU(True),
            nn.Linear(int(t_out*size_factor), t_out),
        )
        self.amptime =  FNOAmpTimeCell(t_in, t_out)
        self.conv = nn.Sequential(nn.Conv2d(dim*t_out, dim*t_out, kernel_size=3,padding=1),
                                  nn.GroupNorm(4, dim*t_out),
                                  nn.SiLU(),
                                  nn.Conv2d(dim*t_out, dim*t_out, kernel_size=3,padding=1),)

    def forward(self, x):
        residual = self.tmlp(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
        x = self.amptime(x)
        x = x + residual

        residual = x
        x = rearrange(x, 'b t c h w -> b (t c) h w')
        x = self.conv(x)
        x = rearrange(x, 'b (t c) h w -> b t c h w', t=self.t_out)
        x = x + residual
        return x
    
class AmpliNet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, dim, hidden_dim, n_layers=3, mlp_ratio=2):
        super().__init__()
        self.pre_seq_length, self.aft_seq_length = pre_seq_length, aft_seq_length
        self.dim, self.hidden_dim = dim, hidden_dim
        self.tmlp = nn.Sequential(
            nn.Linear(pre_seq_length, int(aft_seq_length*mlp_ratio)),
            nn.SELU(True),
            nn.Linear(int(aft_seq_length*mlp_ratio), aft_seq_length),
        )
        self.convin = nn.Sequential(ResnetBlock(dim, hidden_dim),
                                    ResnetBlock(hidden_dim, hidden_dim),
                                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1))
        self.amplist = nn.ModuleList([
            AmpCell(pre_seq_length if i==0 else aft_seq_length, aft_seq_length, hidden_dim) for i in range(n_layers)
        ])
        self.convout = nn.Sequential(ResnetBlock(hidden_dim, hidden_dim),
                                     ResnetBlock(hidden_dim, hidden_dim),
                                     nn.Conv2d(hidden_dim, dim, kernel_size=1))

    def forward(self, x):
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convin(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.pre_seq_length)
        x_ = x.permute(0,2,3,4,1)
        xr = self.tmlp(x_)
        xr = rearrange(xr, 'b c h w t -> (b t) c h w')
        for ampcell in self.amplist:
            x = ampcell(x)
        x = xr + rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.convout(x)
        x = rearrange(x, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        return x
    
class AlphaPre_Amplinet(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, input_shape, input_dim, 
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
        self.criterion = nn.MSELoss()
        self.itr = 0
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate =  self.amp_weight/self.aweight_stop_steps
        
    def forward(self, x, y, cmp_fft_loss=False): # x:[b,t,c,h,w]
        self.itr += 1
        xas = self.amplinet(x)
        xas = torch.sigmoid(xas)
        return xas

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        
        xas = self(frames_in, frames_gt, compute_loss)
        if compute_loss:
            
            loss = 0.
            

            anet_loss = self.criterion(xas, frames_gt)
            loss = {'total_loss': anet_loss}
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
    model = AlphaPre_Amplinet(pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape, input_dim=img_channels, 
                     hidden_dim=dim, n_layers=n_layers, spec_num=spec_num,
                     pha_weight=pha_weight, anet_weight=anet_weight, amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
                     )
    
    return model