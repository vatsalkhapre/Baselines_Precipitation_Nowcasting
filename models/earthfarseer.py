import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from FoTF_module import *
from Temporal_block import *
from utils import *
from timm.models.swin_transformer import SwinTransformerBlock, window_partition, window_reverse
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
from functools import partial
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint_sequential
from torch import nn


class SwinSubBlock(SwinTransformerBlock):
    """A block of Swin Transformer."""

    def __init__(self, dim, input_resolution=None, layer_i=0, mlp_ratio=4., drop=0., drop_path=0.1):
        window_size = 7 if input_resolution[0] % 7 == 0 else max(4, input_resolution[0] // 16)
        window_size = min(8, window_size)
        shift_size = 0 if (layer_i % 2 == 0) else window_size // 2
        super().__init__(dim, input_resolution, num_heads=8, window_size=window_size,
                         shift_size=shift_size, mlp_ratio=mlp_ratio,
                         drop_path=drop_path, drop=drop, qkv_bias=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    def forward(self, x):

        B1, T1, C1, H1, W1 = x.shape # B, 7, 1, 384, 384
        x = x.reshape(B1, T1*C1, H1, W1)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x.reshape(B1, T1, C1, H1, W1) #  7 1 384 384

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=None, patch_size=8, in_c=13, embed_dim=768, norm_layer=None):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1]) # h, w
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.projection= nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Error..."
        '''
        [32, 3, 224, 224] -> [32, 768, 14, 14] -> [32, 768, 196] -> [32, 196, 768]
        Conv2D: [32, 3, 224, 224] -> [32, 768, 14, 14]
        Flatten: [B, C, H, W] -> [B, C, HW]
        Transpose: [B, C, HW] -> [B, HW, C]
        '''
        x = self.projection(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc3 = nn.AdaptiveAvgPool1d(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, F_dim: int, H_dim: int, D: int, gamma: float):

        super().__init__()
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma

        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D)
        )

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)

    def forward(self, x):

        B, N, M = x.shape
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        Y = self.mlp(F)
        PEx = Y.reshape((B, N, self.D))
        return PEx

class AdativeFourierNeuralOperator(nn.Module):
    def __init__(self, dim, h=14, w=14, is_fno_bias=True):
        super(AdativeFourierNeuralOperator, self).__init__()
        self.hidden_size = dim
        self.h = h
        self.w = w
        self.num_blocks = 2
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()
        self.is_fno_bias = is_fno_bias

        if self.is_fno_bias:
            self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        else:
            self.bias = None

        self.softshrink = 0.00

    def multiply(self, input, weights):
        return torch.einsum('...bd, bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape

        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, self.h, self.w, C)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0], inplace=True)
        x_imag = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1], inplace=True)
        x_real = self.multiply(x_real, self.w2[0]) - self.multiply(x_imag, self.w2[1]) + self.b2[0]
        x_imag = self.multiply(x_real, self.w2[1]) + self.multiply(x_imag, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real, x_imag], dim=-1)
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(self.h, self.w), dim=(1,2), norm='ortho')
        x = x.reshape(B, N, C)

        return x+bias

class FourierNetBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=2.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 h=14,
                 w=14):
        super(FourierNetBlock, self).__init__()
        self.normlayer1 = norm_layer(dim)
        self.filter = AdativeFourierNeuralOperator(dim, h=h, w=w)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.normlayer2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.double_skip = True

    def forward(self, x):
        x = x + self.drop_path(self.filter(self.normlayer1(x)))
        x = x + self.drop_path(self.mlp(self.normlayer2(x)))
        return x

class GF_Block(nn.Module):
    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 in_channels=20,
                 out_channels=20,
                 input_frames=20,
                 embed_dim=768,
                 depth=12,
                 mlp_ratio=4.,
                 uniform_drop=False,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 dropcls=0.):
        super(GF_Block, self).__init__()
        self.embed_dim = embed_dim
        self.num_frames = input_frames
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_c=in_channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)) # [1, 196, 768]
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.h = self.patch_embed.grid_size[0]
        self.w = self.patch_embed.grid_size[1]
        '''
        stochastic depth decay rule
        '''
        if uniform_drop:
            dpr = [drop_path_rate for _ in range(depth)]
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            drop_path=dpr[i],
            act_layer=nn.GELU,
            norm_layer=norm_layer,
            h=self.h,
            w=self.w)
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.linearprojection = nn.Sequential(OrderedDict([
            ('transposeconv1', nn.ConvTranspose2d(embed_dim, out_channels * 16, kernel_size=(2, 2), stride=(2, 2))),
            ('act1', nn.Tanh()),
            ('transposeconv2', nn.ConvTranspose2d(out_channels * 16, out_channels * 4, kernel_size=(2, 2), stride=(2, 2))),
            ('act2', nn.Tanh()),
            ('transposeconv3', nn.ConvTranspose2d(out_channels * 4, out_channels, kernel_size=(4, 4), stride=(4, 4)))
        ]))

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        '''
        patch_embed:
        [B, T, C, H, W] -> [B*T, num_patches, embed_dim] L D
        '''
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.patch_embed(x)
        #enc = LearnableFourierPositionalEncoding(768, 768, 64, 768, 10)
       # fourierpos_embed = enc(x)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.pos_drop(x + fourierpos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).transpose(1, 2)
        x = torch.reshape(x, [-1, self.embed_dim, self.h, self.w])
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.linearprojection(x)
        x = x.reshape(B, T, C, H, W)
        return x



class Fourier_Model(nn.Module):
    def __init__(self, shape_in, hid_S=512, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(Fourier_Model, self).__init__()
        self.fourier = GF_Block(img_size=128,
                                patch_size=16,
                                in_channels=2,
                                out_channels=2,
                                input_frames=12,
                                embed_dim=768,
                                depth=12,
                                mlp_ratio=4.,
                                uniform_drop=False,
                                drop_rate=0.,
                                drop_path_rate=0.,
                                norm_layer=None,
                                dropcls=0.)

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        pde_features = self.fourier(x_raw)
        return pde_features
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
        super(BasicConv2d, self).__init__()
        self.act_norm=act_norm
        if not transpose:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=stride //2 )
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
        super(ConvSC, self).__init__()
        if stride == 1:
            transpose = False
        self.conv = BasicConv2d(C_in, C_out, kernel_size=3, stride=stride,
                                padding=1, transpose=transpose, act_norm=act_norm)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
        super(GroupConv2d, self).__init__()
        self.act_norm = act_norm
        if in_channels % groups != 0:
            groups = 1
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class Inception(nn.Module):
    def __init__(self, C_in, C_hid, C_out, incep_ker=[3,5,7,11], groups=8):        
        super(Inception, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

class FoTF(nn.Module):
    def __init__(self, shape_in, num_interactions=3):
        super(FoTF, self).__init__()
        T, C, H, W = shape_in
        self.lc_block = Local_CNN_Branch(in_channels=C, out_channels=C)
        self.gf_block = GF_Block(
            img_size=H,
            patch_size=16,
            in_channels=C,
            out_channels=C,
            input_frames=T,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            uniform_drop=False,
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=None,
            dropcls=0.
        )
        self.up = nn.ConvTranspose2d(C, C, kernel_size=3, stride=1, padding=1)
        self.down = nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1)
        self.conv1x1 = nn.Conv2d(C, C, kernel_size=1)
        self.num_interactions = num_interactions

    def forward(self, x_raw):
        B, T, C, H, W = x_raw.shape
        gf_features = self.gf_block(x_raw)
        lc_features = self.lc_block(x_raw)

        for _ in range(self.num_interactions):
            gf_features_up = self.up(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_up + lc_features

            gf_features = self.gf_block(combined_features)
            lc_features = self.lc_block(combined_features)

            gf_features_down = self.down(gf_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            lc_features = self.conv1x1(lc_features.view(B * T, C, H, W)).view(B, T, C, H, W)
            combined_features = gf_features_down + lc_features

            gf_features = self.gf_block(combined_features)
            lc_features = self.lc_block(combined_features)

        return gf_features + lc_features


class Local_CNN_Branch(nn.Module):
    def __init__(self, in_channels = 2, out_channels = 2):
        super(Local_CNN_Branch, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.upconv = nn.ConvTranspose2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # rearrange dimensions to: (B*T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.upconv(x)
        # return to original dimensions: (B, T, C, H, W)
        x = x.view(B, T, C, x.shape[2], x.shape[3])
        return x
    
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()

    class BasicConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, transpose=False, act_norm=False):
            super(ConvolutionalNetwork.BasicConv2d, self).__init__()
            self.act_norm = act_norm
            if not transpose:
                self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=stride // 2)
            self.norm = nn.GroupNorm(2, out_channels)
            self.act = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            y = self.conv(x)
            if self.act_norm:
                y = self.act(self.norm(y))
            return y

    class ConvSC(nn.Module):
        def __init__(self, C_in, C_out, stride, transpose=False, act_norm=True):
            super(ConvolutionalNetwork.ConvSC, self).__init__()
            if stride == 1:
                transpose = False
            self.conv = ConvolutionalNetwork.BasicConv2d(C_in, C_out, kernel_size=3, stride=stride, padding=1, transpose=transpose, act_norm=act_norm)

        def forward(self, x):
            y = self.conv(x)
            return y

    class GroupConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, act_norm=False):
            super(ConvolutionalNetwork.GroupConv2d, self).__init__()
            self.act_norm = act_norm
            if in_channels % groups != 0:
                groups = 1
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.norm = nn.GroupNorm(groups, out_channels)
            self.activate = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            y = self.conv(x)
            if self.act_norm:
                y = self.activate(self.norm(y))
            return y

    class Inception(nn.Module):
        def __init__(self, C_in, C_hid, C_out, incep_ker=[3, 5, 7, 11], groups=8):
            super(ConvolutionalNetwork.Inception, self).__init__()
            self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)
            layers = []
            for ker in incep_ker:
                layers.append(ConvolutionalNetwork.GroupConv2d(C_hid, C_out, kernel_size=ker, stride=1, padding=ker // 2, groups=groups, act_norm=True))
            self.layers = nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            y = 0
            for layer in self.layers:
                y += layer(x)
            return y

    class Encoder(nn.Module):
        def __init__(self, C_in, C_hid, N_S):
            super(ConvolutionalNetwork.Encoder, self).__init__()
            strides = self.stride_generator(N_S)
            self.enc = nn.Sequential(
                ConvolutionalNetwork.ConvSC(C_in, C_hid, stride=strides[0]),
                *[ConvolutionalNetwork.ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
            )

        def forward(self, x):
            enc1 = self.enc[0](x)
            latent = enc1
            for i in range(1, len(self.enc)):
                latent = self.enc[i](latent)
            return latent, enc1

        @staticmethod
        def stride_generator(N, reverse=False):
            strides = [1, 2] * 10
            if reverse:
                return list(reversed(strides[:N]))
            else:
                return strides[:N]

    class Decoder(nn.Module):
        def __init__(self, C_hid, C_out, N_S):
            super(ConvolutionalNetwork.Decoder, self).__init__()
            strides = ConvolutionalNetwork.Encoder.stride_generator(N_S, reverse=True)
            self.dec = nn.Sequential(
                *[ConvolutionalNetwork.ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
                ConvolutionalNetwork.ConvSC(2 * C_hid, C_hid, stride=strides[-1], transpose=True)
            )
            self.readout = nn.Conv2d(C_hid, C_out, 1)

        def forward(self, hid, enc1=None):
            for i in range(0, len(self.dec) - 1):
                hid = self.dec[i](hid)
            Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
            Y = self.readout(Y)
            return Y

    class Mid_Xnet(nn.Module):
        def __init__(self, channel_in, channel_hid, N_T, incep_ker=[3, 5, 7, 11], groups=8):
            super(ConvolutionalNetwork.Mid_Xnet, self).__init__()

            self.N_T = N_T
            enc_layers = [ConvolutionalNetwork.Inception(channel_in, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
            for i in range(1, N_T - 1):
                enc_layers.append(ConvolutionalNetwork.Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
            enc_layers.append(ConvolutionalNetwork.Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))

            dec_layers = [ConvolutionalNetwork.Inception(channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups)]
            for i in range(1, N_T - 1):
                dec_layers.append(ConvolutionalNetwork.Inception(2 * channel_hid, channel_hid // 2, channel_hid, incep_ker=incep_ker, groups=groups))
            dec_layers.append(ConvolutionalNetwork.Inception(2 * channel_hid, channel_hid // 2, channel_in, incep_ker=incep_ker, groups=groups))

            self.enc = nn.Sequential(*enc_layers)
            self.dec = nn.Sequential(*dec_layers)

        def forward(self, x):
            B, T, C, H, W = x.shape
            x = x.reshape(B, T * C, H, W)

            skips = []
            z = x
            for i in range(self.N_T):
                z = self.enc[i](z)
                if i < self.N_T - 1:
                    skips.append(z)

            z = self.dec[0](z)
            for i in range(1, self.N_T):
                z = self.dec[i](torch.cat([z, skips[-i]], dim=1))

            y = z.reshape(B, T, C, H, W)
            return y

    class skip_connection(nn.Module):
        def __init__(self, shape_in, hid_S=16, hid_T=256, N_S=4, N_T=8, incep_ker=[3, 5, 7, 11], groups=8):
            super(ConvolutionalNetwork.skip_connection, self).__init__()
            T, C, H, W = shape_in
            self.enc = ConvolutionalNetwork.Encoder(C, hid_S, N_S)
            self.hid = ConvolutionalNetwork.Mid_Xnet(T * hid_S, hid_T, N_T, incep_ker, groups)
            self.dec = ConvolutionalNetwork.Decoder(hid_S, C, N_S)

        def forward(self, x_raw):
            B, T, C, H, W = x_raw.shape
            x = x_raw.view(B * T, C, H, W)

            embed, skip = self.enc(x)
            _, C_, H_, W_ = embed.shape

            z = embed.view(B, T, C_, H_, W_)
            hid = self.hid(z)
            hid = hid.reshape(B * T, C_, H_, W_)

            Y = self.dec(hid, skip)
            Y = Y.reshape(B, T, C, H, W)
            return Y




class Earthfarseer_model(nn.Module):
    def __init__(self, shape_in, hid_S=512, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(Earthfarseer_model, self).__init__()
        T, C, H, W = shape_in
        self.H1 = int(H / 2 ** (N_S / 2)) + 1 if H % 3 == 0 else int(H / 2 ** (N_S / 2))
        self.W1 = int(W / 2 ** (N_S / 2))

        self.fotf_encoder = FoTF(shape_in=shape_in)
        self.skip_conneciton = ConvolutionalNetwork.skip_connection(shape_in=shape_in)
        self.latent_projection = Encoder(C, hid_S, N_S)
        self.enc = Encoder(C, hid_S, N_S)
        self.TeDev_block = TeDev(T*hid_S, hid_T, N_T, self.H1, self.W1, incep_ker, groups) #
        self.dec = Decoder(hid_S, C, N_S)


    def forward(self, input_st_tensors):
        # Spatial block FoTF
        B, T, C, H, W = input_st_tensors.shape
        skip_feature = self.skip_conneciton(input_st_tensors)
        spatial_feature = self.fotf_encoder(input_st_tensors)

        spatial_feature = spatial_feature.reshape(-1, C, H, W)
        spatial_embed, spatial_skip_feature = self.latent_projection(spatial_feature)
        _, C_, H_, W_ = spatial_embed.shape # BxT, D h w
        spatial_embed = spatial_embed.view(B, T, C_, H_, W_) # B, T, D ,h, w


        # Temporal block TeDev
        spatialtemporal_embed = self.TeDev_block(spatial_embed)
        spatialtemporal_embed = spatialtemporal_embed.reshape(B*T, C_, H_, W_)


        # Decoder
        predictions = self.dec(spatialtemporal_embed, spatial_skip_feature)
        predictions = predictions.reshape(B, T, C, H, W) + skip_feature
        
        return predictions

if __name__ == '__main__':
    x = torch.randn((1, 10, 1, 64, 64))
    y = torch.randn((1, 10, 1, 64, 64))
    model1 = Earthfarseer_model(shape_in=(10, 1, 64, 64))
    output = model1(x)
    print("input shape:", x.shape)
    print("output shape:", output.shape)

    def model_memory_usage_in_bytes(model):
        total_bytes = 0
        for param in model.parameters():
            num_elements = np.prod(param.data.shape)
            total_bytes += num_elements * 4  
        return total_bytes
    
    total_bytes = model_memory_usage_in_bytes(model1) 
    mb = total_bytes   / 1048576
    print(f'Total memory used by the model parameters: {mb} MB')

class EarthFarseer(nn.Module):
    def __init__(self, T_in, T_out, C, H, W, 
                 hid_S=512, hid_T=256, N_S=4, N_T=8,
                 incep_ker=[3,5,7,11], groups=8):
        super(EarthFarseer, self).__init__()
        
        self.T_in = T_in
        self.T_out = T_out
        self.criterion = nn.MSELoss()
        
        self.model = Earthfarseer_model(
            shape_in=(T_in, C, H, W),
            hid_S=hid_S,
            hid_T=hid_T,
            N_S=N_S,
            N_T=N_T,
            incep_ker=incep_ker,
            groups=groups,
            T_out=T_out
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        pred = self.forward(frames_in)  # (B, T_out, C, H, W)
        
        if compute_loss:
            loss = self.criterion(pred, frames_gt)
            loss = {'total_loss': loss}
            return pred, loss
        else:
            return pred, None


def get_model(
    img_channels=1,
    T_in=10,
    T_out=10,
    input_shape=(128, 128),
    hid_S=512,
    hid_T=256,
    N_S=4,
    N_T=8,
    incep_ker=[3, 5, 7, 11],
    groups=8,
    **kwargs
):
    H, W = input_shape
    model = EarthFarseer(
        T_in=T_in,
        T_out=T_out,
        C=img_channels,
        H=H,
        W=W,
        hid_S=hid_S,
        hid_T=hid_T,
        N_S=N_S,
        N_T=N_T,
        incep_ker=incep_ker,
        groups=groups
    )
    return model