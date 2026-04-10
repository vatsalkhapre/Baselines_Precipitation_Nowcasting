import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from einops import rearrange

class OrderedEasyDict(OrderedDict):
    """Using OrderedDict for the `easydict` package
    See Also https://pypi.python.org/pypi/easydict/
    """
    def __init__(self, d=None, **kwargs):
        super(OrderedEasyDict, self).__init__()
        if d is None:
            d = OrderedDict()
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        # special handling of self.__root and self.__map
        if name.startswith('_') and (name.endswith('__root') or name.endswith('__map')):
            super(OrderedEasyDict, self).__setattr__(name, value)
        else:
            if isinstance(value, (list, tuple)):
                value = [self.__class__(x)
                         if isinstance(x, dict) else x for x in value]
            else:
                value = self.__class__(value) if isinstance(value, dict) else value
            super(OrderedEasyDict, self).__setattr__(name, value)
            super(OrderedEasyDict, self).__setitem__(name, value)

# input: B, C, H, W
# flow: [B, 2, H, W]
class activation():

    def __init__(self, act_type, negative_slope=0.2, inplace=True):
        super().__init__()
        self._act_type = act_type
        self.negative_slope = negative_slope
        self.inplace = inplace

    def __call__(self, input):
        if self._act_type == 'leaky':
            return F.leaky_relu(input, negative_slope=self.negative_slope, inplace=self.inplace)
        elif self._act_type == 'relu':
            return F.relu(input, inplace=self.inplace)
        elif self._act_type == 'sigmoid':
            return torch.sigmoid(input)
        else:
            raise NotImplementedError
        
__C = OrderedEasyDict()
cfg = __C

# --- ADD THIS BLOCK TO PREVENT CRASHES ---
cfg.GLOBAL = OrderedEasyDict()
# Automatically use GPU if available
cfg.GLOBAL.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

cfg.MODEL = OrderedEasyDict()
# TrajGRU usually uses leaky relu for its internal flow generation
cfg.MODEL.RNN_ACT_TYPE = activation('leaky') 

cfg.HKO = OrderedEasyDict()
cfg.HKO.BENCHMARK = OrderedEasyDict()
# Default input length fallback
cfg.HKO.BENCHMARK.IN_LEN = 5 
# -----------------------------------------


        
def wrap(input, flow):
    B, C, H, W = input.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(cfg.GLOBAL.DEVICE)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(cfg.GLOBAL.DEVICE)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(input, vgrid)
    return output

class BaseConvRNN(nn.Module):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type=torch.tanh,
                 prefix='BaseConvRNN'):
        super(BaseConvRNN, self).__init__()
        self._prefix = prefix
        self._num_filter = num_filter
        self._h2h_kernel = h2h_kernel
        assert (self._h2h_kernel[0] % 2 == 1) and (self._h2h_kernel[1] % 2 == 1), \
            "Only support odd number, get h2h_kernel= %s" % str(h2h_kernel)
        self._h2h_pad = (h2h_dilate[0] * (h2h_kernel[0] - 1) // 2,
                         h2h_dilate[1] * (h2h_kernel[1] - 1) // 2)
        self._h2h_dilate = h2h_dilate
        self._i2h_kernel = i2h_kernel
        self._i2h_stride = i2h_stride
        self._i2h_pad = i2h_pad
        self._i2h_dilate = i2h_dilate
        self._act_type = act_type
        assert len(b_h_w) == 3
        i2h_dilate_ksize_h = 1 + (self._i2h_kernel[0] - 1) * self._i2h_dilate[0]
        i2h_dilate_ksize_w = 1 + (self._i2h_kernel[1] - 1) * self._i2h_dilate[1]
        self._batch_size, self._height, self._width = b_h_w
        self._state_height = (self._height + 2 * self._i2h_pad[0] - i2h_dilate_ksize_h)\
                             // self._i2h_stride[0] + 1
        self._state_width = (self._width + 2 * self._i2h_pad[1] - i2h_dilate_ksize_w) \
                             // self._i2h_stride[1] + 1
        self._curr_states = None
        self._counter = 0


class TrajGRU(BaseConvRNN):
    # b_h_w: input feature map size
    def __init__(self, input_channel, num_filter, b_h_w, zoneout=0.0, L=5,
                 i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                 h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                 act_type=cfg.MODEL.RNN_ACT_TYPE):
        super(TrajGRU, self).__init__(num_filter=num_filter,
                                      b_h_w=b_h_w,
                                      h2h_kernel=h2h_kernel,
                                      h2h_dilate=h2h_dilate,
                                      i2h_kernel=i2h_kernel,
                                      i2h_pad=i2h_pad,
                                      i2h_stride=i2h_stride,
                                      act_type=act_type,
                                      prefix='TrajGRU')
        self._L = L
        self._zoneout = zoneout

        # 对应 wxz, wxr, wxh
        # reset_gate, update_gate, new_mem
        self.i2h = nn.Conv2d(in_channels=input_channel,
                            out_channels=self._num_filter*3,
                            kernel_size=self._i2h_kernel,
                            stride=self._i2h_stride,
                            padding=self._i2h_pad,
                            dilation=self._i2h_dilate)

        # inputs to flow
        self.i2f_conv1 = nn.Conv2d(in_channels=input_channel,
                                out_channels=32,
                                kernel_size=(5, 5),
                                stride=1,
                                padding=(2, 2),
                                dilation=(1, 1))

        # hidden to flow
        self.h2f_conv1 = nn.Conv2d(in_channels=self._num_filter,
                                   out_channels=32,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2),
                                   dilation=(1, 1))

        # generate flow
        self.flows_conv = nn.Conv2d(in_channels=32,
                                   out_channels=self._L * 2,
                                   kernel_size=(5, 5),
                                   stride=1,
                                   padding=(2, 2))



        self.ret = nn.Conv2d(in_channels=self._num_filter*self._L,
                                   out_channels=self._num_filter*3,
                                   kernel_size=(1, 1),
                                   stride=1)



    # inputs: B*C*H*W
    def _flow_generator(self, inputs, states):
        if inputs is not None:
            i2f_conv1 = self.i2f_conv1(inputs)
        else:
            i2f_conv1 = None
        h2f_conv1 = self.h2f_conv1(states)
        f_conv1 = i2f_conv1 + h2f_conv1 if i2f_conv1 is not None else h2f_conv1
        f_conv1 = self._act_type(f_conv1)

        flows = self.flows_conv(f_conv1)
        flows = torch.split(flows, 2, dim=1)
        return flows

    # inputs: S*B*C*H*W
    def forward(self, inputs=None, states=None, seq_len=cfg.HKO.BENCHMARK.IN_LEN):
        if states is None:
            states = torch.zeros((inputs.size(1), self._num_filter, self._state_height,
                                  self._state_width), dtype=torch.float).to(cfg.GLOBAL.DEVICE)
        if inputs is not None:
            S, B, C, H, W = inputs.size()
            i2h = self.i2h(torch.reshape(inputs, (-1, C, H, W)))
            i2h = torch.reshape(i2h, (S, B, i2h.size(1), i2h.size(2), i2h.size(3)))
            i2h_slice = torch.split(i2h, self._num_filter, dim=2)

        else:
            i2h_slice = None

        prev_h = states
        outputs = []
        for i in range(seq_len):
            if inputs is not None:
                flows = self._flow_generator(inputs[i, ...], prev_h)
            else:
                flows = self._flow_generator(None, prev_h)
            wrapped_data = []
            for j in range(len(flows)):
                flow = flows[j]
                wrapped_data.append(wrap(prev_h, -flow))
            wrapped_data = torch.cat(wrapped_data, dim=1)
            h2h = self.ret(wrapped_data)
            h2h_slice = torch.split(h2h, self._num_filter, dim=1)
            if i2h_slice is not None:
                reset_gate = torch.sigmoid(i2h_slice[0][i, ...] + h2h_slice[0])
                update_gate = torch.sigmoid(i2h_slice[1][i, ...] + h2h_slice[1])
                new_mem = self._act_type(i2h_slice[2][i, ...] + reset_gate * h2h_slice[2])
            else:
                reset_gate = torch.sigmoid(h2h_slice[0])
                update_gate = torch.sigmoid(h2h_slice[1])
                new_mem = self._act_type(reset_gate * h2h_slice[2])
            next_h = update_gate * prev_h + (1 - update_gate) * new_mem
            if self._zoneout > 0.0:
                mask = F.dropout2d(torch.zeros_like(prev_h), p=self._zoneout)
                next_h = torch.where(mask, next_h, prev_h)
            outputs.append(next_h)
            prev_h = next_h

        # return torch.cat(outputs), next_h
        return torch.stack(outputs), next_h
    
class TrajGRUEncoder_128(nn.Module):
    def __init__(self, batch_size=2):
        super(TrajGRUEncoder_128, self).__init__()
        
        # Modified: econv1 (In: 128x128 -> Out: 64x64)
        self.econv1 = SequenceConv(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1),
            activation=nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Modified: ernn1 (In Res: 64x64)
        self.ernn1 = TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 64, 64), L=13)
        
        # Modified: edown1 (In: 64x64 -> Out: 32x32)
        self.edown1 = SequenceConv(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            activation=nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ernn2 remains on 32x32
        self.ernn2 = TrajGRU(input_channel=64, num_filter=192, b_h_w=(batch_size, 32, 32), L=13)
        
        # edown2 remains identical (In: 32x32 -> Out: 16x16)
        self.edown2 = SequenceConv(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1),
            activation=nn.LeakyReLU(0.2, inplace=True)
        )
        
        # ernn3 remains on 16x16
        self.ernn3 = TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), L=9)

    def forward(self, inputs):
        S = inputs.size(0)
        c1 = self.econv1(inputs)
        r1_seq, state1 = self.ernn1(inputs=c1, seq_len=S)
        
        d1 = self.edown1(r1_seq)
        r2_seq, state2 = self.ernn2(inputs=d1, seq_len=S)
        
        d2 = self.edown2(r2_seq)
        r3_seq, state3 = self.ernn3(inputs=d2, seq_len=S)
        
        return state1, state2, state3

class SequenceConv(nn.Module):
    """
    Helper module to apply a 2D Convolution or Deconvolution over a 5D sequence tensor.
    Input shape: [S, B, C, H, W] -> Output shape: [S, B, C_out, H_out, W_out]
    """
    def __init__(self, conv_layer, activation=None):
        super(SequenceConv, self).__init__()
        self.conv = conv_layer
        self.activation = activation

    def forward(self, x):
        S, B, C, H, W = x.size()
        x_flat = x.reshape(S * B, C, H, W)
        out_flat = self.conv(x_flat)
        if self.activation is not None:
            out_flat = self.activation(out_flat)
        _, C_out, H_out, W_out = out_flat.size()
        return out_flat.view(S, B, C_out, H_out, W_out)
    
class TrajGRUForecaster_128(nn.Module):
    def __init__(self, batch_size=2):
        super(TrajGRUForecaster_128, self).__init__()
        
        # frnn1 remains on 16x16
        self.frnn1 = TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), L=9)
        
        # Modified: fup1 (In: 16x16 -> Out: 32x32)
        self.fup1 = SequenceConv(
            nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=4, stride=2, padding=1),
            activation=nn.LeakyReLU(0.2, inplace=True)
        )
        
        # frnn2 remains on 32x32
        self.frnn2 = TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), L=13)
        
        # Modified: fup2 (In: 32x32 -> Out: 64x64)
        self.fup2 = SequenceConv(
            nn.ConvTranspose2d(in_channels=192, out_channels=192, kernel_size=4, stride=2, padding=1),
            activation=nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Modified: frnn3 (In Res: 64x64)
        self.frnn3 = TrajGRU(input_channel=192, num_filter=64, b_h_w=(batch_size, 64, 64), L=13)
        
        # Modified: fdeconv4 (In: 64x64 -> Out: 128x128)
        self.fdeconv4 = SequenceConv(
            nn.ConvTranspose2d(in_channels=64, out_channels=8, kernel_size=4, stride=2, padding=1),
            activation=nn.LeakyReLU(0.2, inplace=True)
        )
        
        # fconv5 remains identical (In: 128x128 -> Out: 128x128)
        self.fconv5 = SequenceConv(
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0),
            activation=None
        )

    def forward(self, enc_states, future_seq_len):
        state1, state2, state3 = enc_states
        
        r1_seq, _ = self.frnn1(inputs=None, states=state3, seq_len=future_seq_len)
        
        u1 = self.fup1(r1_seq)
        r2_seq, _ = self.frnn2(inputs=u1, states=state2, seq_len=future_seq_len)
        
        u2 = self.fup2(r2_seq)
        r3_seq, _ = self.frnn3(inputs=u2, states=state1, seq_len=future_seq_len)
        
        d4 = self.fdeconv4(r3_seq)
        final_predictions = self.fconv5(d4)
        
        return final_predictions

class HKO7_TrajGRU_Pipeline(nn.Module):
    def __init__(self, future_seq_len, batch_size=2):
        super(HKO7_TrajGRU_Pipeline, self).__init__()
        self.encoder = TrajGRUEncoder_128(batch_size=batch_size)
        self.forecaster = TrajGRUForecaster_128(batch_size=batch_size)
        self.future_seq_len = future_seq_len

    def forward(self, past_inputs):
        """
        past_inputs: [Input_Seq_Len, Batch, 1, 480, 480]
        future_seq_len: Int (default 20 as per HKO-7 benchmark)
        """
        # Encode the past 5 frames
        enc_states = self.encoder(past_inputs)
        
        # Forecast the next 20 frames
        predictions = self.forecaster(enc_states, self.future_seq_len)
        
        return predictions
    
class TrajGRU_model(nn.Module):
    def __init__(self, future_seq_len, batch_size, **kwargs) -> None:
        super().__init__()
        self.net = HKO7_TrajGRU_Pipeline(future_seq_len, batch_size)
        self.criterion = nn.MSELoss()
    
    def forward(self, x):
        x = rearrange(x, 'b t c h w -> t b c h w')
        out = self.net(x)
        out = rearrange(out, 't b c h w -> b t c h w')
        return out
    
    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        B = frames_in.shape[0]
        out = self(frames_in)
        pred = out
        if compute_loss:
            loss = self.criterion(out, frames_gt)
            return pred, loss
        else:
            return pred, None