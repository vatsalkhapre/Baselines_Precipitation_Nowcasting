import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv3d

torch.manual_seed(0)
np.random.seed(0)

# %%
""" The forward operation """
class WNO3d(nn.Module):
    def __init__(self, in_channels, level, layers, size, wavelet):
        super(WNO3d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 4-channel tensor, Input at t0 and location (a(x,y,t), t, x, y)
              : shape: (batchsize * t=time * x=width * x=height * c=4)
        Output: Solution of a later timestep (u(x, T_in+1))
              : shape: (batchsize * t=time * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition, initially 1
        layers: scalar, number of wavelet kernel integral blocks initially start with 1 and then 2
        size  : list with 3 elements (for 3D), the 3D volume size, should match the size of input tensor. 
        wavelet   : string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 3 elements (for 3D), right supports of the 3D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = in_channels
        self.size = size
        self.layers = layers
                
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
     
        for i in range(self.layers):
            self.conv.append(WaveConv3d(self.width, self.width, self.level, 
                                        self.size, wavelet))
            self.w.append(nn.Conv3d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, self.width)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.fc0(x)                 # Shape: Batch * x * y * z * Channel
        # x = x.permute(0, 4, 3, 1, 2)    # Shape: Batch * Channel * z * x * y 
        # if self.padding != 0:
        #     x = F.pad(x, [0,self.padding, 0,self.padding, 0,self.padding]) # do padding, if required
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
            
        # if self.padding != 0:
        #     x = x[..., :-self.padding, :-self.padding, :-self.padding] # remove padding, when required
        x = x.permute(0, 3, 4, 2, 1)        # Shape: Batch * x * y * z * Channel 
        x = self.fc2(F.mish(self.fc1(x)))   # Shape: Batch * x * y * z 
        return x
    
