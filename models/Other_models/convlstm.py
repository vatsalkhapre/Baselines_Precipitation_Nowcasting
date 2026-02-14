import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()


        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        assert len(hidden_dim) == num_layers and len(kernel_size) == num_layers

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, seq_len, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=(h, c))
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class EncoderForecaster(nn.Module):
    """
    Encoder-Forecaster structure following the paper:
    - encoder consumes J input frames and returns last hidden/cell states
    - forecaster unfolds K timesteps, initialized with encoder last states
    - at each decode step gather hidden maps from all forecaster layers, concat channels
    and apply 1x1 conv to produce logits for that timestep
    """
    def __init__(self, input_dim=1, hidden_dims=[64, 64], kernel_size=(3, 3), num_layers=2):
        super().__init__()
        assert len(hidden_dims) == num_layers
        self.encoder = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dims,kernel_size=kernel_size, num_layers=num_layers,batch_first=True, 
                                bias=True, return_all_layers=True)
        
        
        self.forecaster = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dims,kernel_size=kernel_size, num_layers=num_layers,batch_first=True,
                                   bias=True, return_all_layers=False)
        
         
        total_channels = sum(hidden_dims)
        self.conv1x1 = nn.Conv2d(in_channels=total_channels, out_channels=input_dim, kernel_size=1, padding=0)
        self.hidden_dims = hidden_dims


    def forward(self, input_seq, pred_len=15, teacher_forcing_ratio=0.0, target_seq=None):
        # input_seq: (B, J, C, H, W)
        device = input_seq.device
        B, J, C, H, W = input_seq.shape


        _, encoder_states = self.encoder(input_seq) # list of (h,c)
        hidden = [(s[0].to(device), s[1].to(device)) for s in encoder_states]
        
        
        zero_input = torch.zeros((B, C, H, W), device=device)
        outputs = []
        for t in range(pred_len):
            if (teacher_forcing_ratio > 0.0) and (target_seq is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                cur_in = target_seq[:, t]
            else:
                cur_in = zero_input
            x = cur_in
            next_hidden = []
            for layer_idx, cell in enumerate(self.forecaster.cell_list):
                
                h, c = hidden[layer_idx]
                h, c = cell(x, (h, c))
                next_hidden.append((h, c))
                x = h
            hidden = next_hidden
            hs = [hidden[i][0] for i in range(len(hidden))]
            concat_h = torch.cat(hs, dim=1)
            logit = self.conv1x1(concat_h)
            outputs.append(logit.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs


class PaperModel(nn.Module):
    """
    Lightweight wrapper exposing .predict(frames_in, frames_gt, compute_loss=True)
    to match the project Runner API. Internally uses EncoderForecaster.
    """
    def __init__(self, frames_in, frames_out, input_channels=1, hidden_dims=[64, 64], kernel_size=(3, 3)):
        super().__init__()
        self.frames_in = frames_in
        self.frames_out = frames_out
        self.net = EncoderForecaster(input_dim=input_channels, hidden_dims=hidden_dims,
        kernel_size=kernel_size, num_layers=len(hidden_dims))
        # loss used in paper: cross-entropy per pixel -> BCEWithLogitsLoss (targets in {0,1} or normalized P)
        self.criterion = nn.BCEWithLogitsLoss()


    def forward(self, x):
        # x: (B, T_in, C, H, W)
        return self.net(x, pred_len=self.frames_out)


    def predict(self, frames_in, frames_gt=None, compute_loss=True):
        # frames_in: (B, T_in, C, H, W)
        device = frames_in.device
        logits = self.forward(frames_in) # (B, T_out, C, H, W)
        preds = torch.sigmoid(logits)
        if not compute_loss or frames_gt is None:
            return preds, None
        # ensure frames_gt shape matches logits
        if frames_gt.shape != logits.shape:
            raise ValueError(f"frames_gt shape {frames_gt.shape} doesn't match logits shape {logits.shape}")
        loss_val = self.criterion(logits, frames_gt)
        return preds, {'total_loss': loss_val}