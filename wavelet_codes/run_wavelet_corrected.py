import os
import os.path as osp
import math
import time
import argparse
import logging 
import yaml
from tqdm import tqdm
from datetime import timedelta
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm, BoundaryNorm
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from datasets.dataset_mosdac import *
from datasets.get_datasets import get_dataset
from utils.tools import print_log, cycle, show_img_info
from copy import deepcopy
from pytorch_wavelets import DWTForward
from pytorch_wavelets import DWTInverse

# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "6427ba1f8d0c13065720163c3aed0fa974031bef"
# os.environ["WANDB_SILENT"] = "true"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"


def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone_L',       type=str,   default='fnoamplinet_mseonly',        help='backbone model for deterministic LL wavelet prediction (alphapre/convlstm_paper/simvp)')
    parser.add_argument('--backbone_H',       type=str,   default='amplinet_latent_mseonly',        help='backbone model for deterministic High frequency wavelet prediction (alphapre/convlstm_paper/simvp)')
    parser.add_argument("--seed",           type=int,   default=0,                 help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='shanghai_unnormalize',      help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default="Shanghai_wavelet_full_corrected_FNOampli_L_ampli_H",              help="additional note for experiment")
    # --------------- Dataset ---------------
    parser.add_argument("--dataset",            type=str,       default='shanghai_unnormalize',   help="dataset name")
    parser.add_argument("--datatype",           type=str,       default='vil_vip',           help="Indicates the datatype available")
    parser.add_argument("--file_rain_seq_add",  type=str,       default=0,              help="Rainy days file")
    parser.add_argument("--method",             type= int,      default= None,          help = "Method to select the dataset as per the need. (Look at the function for more details)")
    parser.add_argument("--img_size",           type=int,       default=64,            help="image size")
    parser.add_argument("--stride",             type=int,       default=13,             help="dataset stride")
    parser.add_argument("--img_channel_L",        type=int,       default=1,              help="channel of image for LL")
    parser.add_argument("--img_channel_H",        type=int,       default=3,              help="channel of image for HF")
    parser.add_argument("--patch",              type=int,       default=2,              help="patch size")
    parser.add_argument("--seq_len",            type=int,       default=25,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",          type=int,       default=5,              help="number of frames to input")
    parser.add_argument("--frames_out",         type=int,       default=20,             help="number of frames to output")    
    parser.add_argument("--num_workers",        type=int,       default=8,              help="number of workers for data loader")
    parser.add_argument("--preprocessing",      type=int,       default=0,              help="Preprocessing 0 for min max normalization")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=1e-4,            help="learning rate")
    parser.add_argument("--lr_beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr_beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=0.0,             help="l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='cosine',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps_L",   type=int,   default=1000,            help="warmup steps for LL model")
    parser.add_argument("--warmup_steps_H",   type=int,   default=1000,            help="warmup steps for HF model")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=8,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=4,               help="batch size")
    parser.add_argument("--epochs",         type=int,   default=100,              help="number of epochs")
    parser.add_argument("--training_steps", type=int,   default=1,               help="number of training steps")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default=None,            help="resumed checkpoint milestone")
    parser.add_argument("--spec_num",       type=int,   default=20,              help="spectral number")
    parser.add_argument("--layers",         type=int,   default=3,               help="layers number")
    parser.add_argument("--pha_weight",     type=float, default=0.01,            help="phase weight")
    parser.add_argument("--amp_weight",     type=float, default=0.01,            help="amplitute weight")
    parser.add_argument("--anet_weight",    type=float, default=0.1,             help="amplitute network mse weight")
    parser.add_argument("--aw_stop_step",   type=int,   default=5000,            help="training step at which the amplitude weight decays to 0")
    parser.add_argument("--out_weight",     type=float, default=1.0,             help="final output weight")
    parser.add_argument("--tf",             action="store_false",                help="teacher force")
    parser.add_argument("--tf_stop_iter",     type=int,     default=2000,        help="teacher force stop iters")
    parser.add_argument("--tf_changing_rate", type=float,   default=0.,          help="teacher force changing rate")
    
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",                 help="evaluation mode")
    parser.add_argument("--valid",          action="store_true",                 help="valid mode")
    parser.add_argument("--valid_limit",    action="store_true",                 help="valid limit mode")
    parser.add_argument("--vlnum",          type=int,   default=30,              help="valid limit nums")
    parser.add_argument("--visual",         action="store_true",                 help="save all test sample visualization")
    parser.add_argument("--gpu_use",        type=str,   nargs='+', default=["0",],  help="gpu(s) to use")
    parser.add_argument("--res_opt",        action="store_true",                 help="resume opt")

    # --------------- Wandb ---------------
    parser.add_argument("--wandb_state",    type=str,   default='online',      help="wandb state config")
    parser.add_argument("--wandb_project_name", type=str, default="Amplinet_wavelet", help="wandb project name")
    parser.add_argument("--run_name",       type=str,   default='Training_waveletFNOampli_L_ampli_H',        help="wandb run name")

    #------------------------- Plots -----------------------------
    parser.add_argument("--generate_outputs", action="store_true",               help="Generate visualizations from checkpoint")
    parser.add_argument("--plot_saving_directory", type=str,  default=None,      help="Enter saving directory for plots")

    args = parser.parse_args()
    return args


class WaveletFeatureExtractor(nn.Module):
    def __init__(self, wave='haar'):
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wave, mode='zero')

    def forward(self, x): 
        """
        x: (B, T, C, H, W)
        Returns:
            yl: (B*T, C, H/2, W/2) - Low-Low component
            yh_cat: (B*T, 3*C, H/2, W/2) - Concatenated LH, HL, HH components
        """
        B, T, C, H, W = x.shape
        x_collapsed = x.reshape(B * T, C, H, W)
        
        device = x.device
        yl, yh = self.dwt(x_collapsed.cpu())
        
        # yh is list: [tensor(B*T, C, 3, H/2, W/2)]
        # The '3' dimension contains [LH, HL, HH]
        yh_tensor = yh[0]  # (B*T, C, 3, H/2, W/2)
        
        # Reshape to (B*T, 3*C, H/2, W/2)
        BT, C_dim, three, Hh, Wh = yh_tensor.shape
        yh_cat = yh_tensor.permute(0, 2, 1, 3, 4).reshape(BT, three * C_dim, Hh, Wh)
        
        return yl.to(device), yh_cat.to(device)


def normalize(dataset_name, value):
    if dataset_name == 'shanghai_unnormalize':
        value = value / 510
        return value

def unnormalize_LL(dataset_name, value):
    """Reverse the normalization applied during training"""
    if dataset_name == 'shanghai_unnormalize':
        value = value * 510  # Reverse the /510 normalization
        return value
    return value

class WaveletDualModel(nn.Module):
    """Container with separate optimization paths"""
    def __init__(self, model_ll, model_hf):
        super().__init__()
        self.model_ll = model_ll
        self.model_hf = model_hf
    
    def predict(self, frames_in_ll, frames_in_hf, frames_gt_ll=None, frames_gt_hf=None, compute_loss=False):
        pred_ll, loss_ll = self.model_ll.predict(
            frames_in=frames_in_ll,
            frames_gt=frames_gt_ll,
            compute_loss=compute_loss
        )
        
        pred_hf, loss_hf = self.model_hf.predict(
            frames_in=frames_in_hf,
            frames_gt=frames_gt_hf,
            compute_loss=compute_loss
        )
        
        if compute_loss:
            total_loss = {
                'total_loss': loss_ll['total_loss'] + loss_hf['total_loss'],
                'll_loss_dict': loss_ll,
                'hf_loss_dict': loss_hf,
            }
            return (pred_ll, pred_hf), total_loss
        else:
            return (pred_ll, pred_hf), None

        
class Runner(object):
    
    def __init__(self, args):
        
        self.args = args
        self.ae_ckpt = args.ae_ckpt_path
        self._preparation()
        self.max_csi, self.best_step = 0.0, 0
        
        # Config DDP kwargs from accelerate
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=self.log_path
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        process_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        
        self.accelerator = Accelerator(
            project_config  =   project_config,
            kwargs_handlers =   [ddp_kwargs, process_kwargs],
            mixed_precision =   self.args.mixed_precision,
            log_with        =   'wandb'
        )
        
        # Config log tracker 'wandb' from accelerate
        self.accelerator.init_trackers(
            project_name=self.args.wandb_project_name,
            config=self.args.__dict__,
            init_kwargs={"wandb": 
                {
                "mode": self.args.wandb_state,
                "name": self.args.run_name
                }
            }
        )
        
        print_log(f"Using GPUs: {self.device}", self.is_main)
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
        print_log(self.accelerator.state, self.is_main)
        
        self._load_data()
        self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.train_loader, self.valid_loader, self.test_loader
        )
        
        # ✅ STEP 1: Build models FIRST
        self._build_model()
        
        # ✅ STEP 2: Create wrapper (models now exist!)
        self.wavelet_dual_model = WaveletDualModel(
            self.model["model_ll"],
            self.model["model_hf"]
        )
        
        # ✅ STEP 3: Build optimizers
        self._build_optimizer()
        
        # ✅ STEP 4: Prepare everything
        self.wavelet_dual_model = self.accelerator.prepare(self.wavelet_dual_model)
        self.optimizer_ll = self.accelerator.prepare(self.optimizer_ll)
        self.optimizer_hf = self.accelerator.prepare(self.optimizer_hf)
        self.scheduler_ll = self.accelerator.prepare(self.scheduler_ll)
        self.scheduler_hf = self.accelerator.prepare(self.scheduler_hf)
        
        # ✅ STEP 5: Create EMA (after prepare)
        self.ema = EMA(self.wavelet_dual_model, beta=self.args.ema_rate, update_every=20).to(self.device)
        
        # ✅ STEP 6: Setup wavelets
        self.dwt = WaveletFeatureExtractor()
        self.idwt = DWTInverse(mode='zero')
        
        self.train_dl_cycle = cycle(self.train_loader)

        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)
            
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)

        if self.args.dataset == 'cikm':
            self.args.frames_in = 5
            self.args.frames_out = 10
        else:
            self.args.frames_in = 5
            self.args.frames_out = 20

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
    
    def _preparation(self):
        set_seed(self.args.seed)
        
        self.exp_name   = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.exp_dir = osp.join(cur_dir, 'Exps', self.args.exp_dir, self.exp_name)        
        self.ckpt_path = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path = osp.join(self.exp_dir, 'test_samples')
        self.log_path = osp.join(self.exp_dir, 'logs')
        self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
        
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        exp_params = self.args.__dict__
        params_path = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'log.log')),
            ]
        )
        
    def _load_data(self):
        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            img_size=self.args.img_size*2,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
            stride=self.args.stride,
            file_rain_seq_add=self.args.file_rain_seq_add,
            method = self.args.method,
            in_channels = self.args.frames_in,
            out_channels = self.args.frames_out,
            preprocess_type = self.args.preprocessing
        )
        
        self.visiual_save_fn = color_save_fn
        self.thresholds = THRESHOLDS
        self.scale_value = PIXEL_SCALE
        
        if self.args.dataset == 'vil_mosdac' or self.args.dataset == 'vil' or self.args.dataset == 'mosdac':
            self.train_loader = create_loader(train_data, batch_size= self.args.batch_size, shuffle=True)
            self.valid_loader = create_loader(valid_data, batch_size= self.args.batch_size)
            self.test_loader = create_loader(test_data, batch_size= self.args.batch_size)

        if self.args.dataset == 'sevir':
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
        else: 
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )

        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",
                  self.is_main)
        
        for sample in self.train_loader:
            print("Sample shape", sample.shape)
            break

        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}", self.is_main)
        print_log(f"Shape of input to the model: {self.args.img_size}x{self.args.img_size}", self.is_main)


    def _build_model(self):
        print_log("Build Model!", self.is_main)
        self.model = nn.ModuleDict()
        
        backbones = [self.args.backbone_L, self.args.backbone_H]
        models = ["model_ll", "model_hf"]
        channels = [self.args.img_channel_L, self.args.img_channel_H]
        
        for i, backbone in enumerate(backbones):
            img_channel = channels[i]
            
            if backbone == 'simvp':
                from models.simvp import get_model
                kwargs = {
                    "in_shape": (img_channel, self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                }
                model = get_model(**kwargs)

            elif backbone == 'alphapre':
                from models.alphapre import get_model
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels': img_channel,
                    'dim': 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)
            
            elif backbone == 'amplinet':
                from models.alphapre_amplinet import get_model
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels': img_channel,
                    'dim': 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)

            elif backbone == 'amplinet_mseonly':
                from models.alphapre_amplinet_MSE_only import get_model
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels': img_channel,
                    'dim': 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)

            elif backbone == 'amplinet_latent_mseonly':
                from models.alphapre_amplinet_MSE_only_latent import get_model
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels' : img_channel,
                    'dim' : 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)

            elif backbone == 'fnoamplinet_mseonly':
                from models.alphapre_fnoamplinet_MSE_only import get_model
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels': img_channel,
                    'dim': 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)
            
            elif backbone == 'afnoamplinet_mseonly':
                from models.alphapre_AFNOamplinet_MSE_only import get_model
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels': img_channel,
                    'dim': 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)

            elif backbone == 'alphapre_amplinet_amp_loss':
                from models.alphapre_amplinet_amp_loss import get_model
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels': img_channel,
                    'dim': 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)

            elif backbone == 'alphapre_phase_net':
                from models.Other_models.alphapre_phasenet import get_model 
                kwargs = {
                    "input_shape": (self.args.img_size, self.args.img_size),
                    "T_in": self.args.frames_in,
                    "T_out": self.args.frames_out,
                    'img_channels': img_channel,
                    'dim': 64,
                    'n_layers': self.args.layers,
                    'pha_weight': self.args.pha_weight,
                    'anet_weight': self.args.anet_weight,
                    'amp_weight': self.args.amp_weight,
                    'spec_num': self.args.spec_num,
                    'aweight_stop_steps': self.args.aw_stop_step,
                }
                model = get_model(**kwargs)

            elif backbone == 'convlstm_paper':
                from models.Other_models.convlstm import PaperModel
                hidden_dims = [64, 64]
                model = PaperModel(frames_in=self.args.frames_in, frames_out=self.args.frames_out,
                input_channels=img_channel, hidden_dims=hidden_dims, kernel_size=(3,3))
            else:
                raise NotImplementedError
            
            self.model[models[i]] = model
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)


    def _build_optimizer(self):
        num_steps_per_epoch = len(self.train_loader)
        self.global_epochs = self.args.epochs
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch
        
        self.cur_step, self.cur_epoch = 0, 0

        warmup_steps_L = self.args.warmup_steps_L
        warmup_steps_H = self.args.warmup_steps_H

        self.optimizer_ll = torch.optim.AdamW(
            self.model["model_ll"].parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=1e-4
        )
        
        self.optimizer_hf = torch.optim.AdamW(
            self.model["model_hf"].parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.95),
            weight_decay=1e-4
        )
        
        self.scheduler_ll = get_cosine_schedule_with_warmup(
            self.optimizer_ll,
            num_warmup_steps=warmup_steps_L,
            num_training_steps=self.global_steps,
        )
        
        self.scheduler_hf = get_cosine_schedule_with_warmup(
            self.optimizer_hf,
            num_warmup_steps=warmup_steps_H,
            num_training_steps=self.global_steps,
        )
        
        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"    LL optimizer lr: {self.args.lr}")
            print_log(f"    HF optimizer lr: {self.args.lr }")
    
    def save(self, svname=None):
        if not self.is_main:
            return
        
        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.wavelet_dual_model),
            'ema': self.ema.state_dict(),
            'opt_ll': self.optimizer_ll.state_dict(),
            'opt_hf': self.optimizer_hf.state_dict(),
            'scheduler_ll': self.scheduler_ll.state_dict(),
            'scheduler_hf': self.scheduler_hf.state_dict(),
        }
        
        if svname == None:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt"))
            print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        else:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{svname}.pt"))
            print_log(f"Save {svname} checkpoint to {self.ckpt_path}", self.is_main)

        
    def load(self, milestone):
        device = self.accelerator.device
        
        if isinstance(milestone, str) and '.pt' in milestone:
            data = torch.load(milestone, map_location=device)
            print_log(f"Load checkpoint {milestone}.", self.is_main)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
            print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
        model = self.accelerator.unwrap_model(self.wavelet_dual_model)
        model.load_state_dict(data['model'])
        self.wavelet_dual_model = self.accelerator.prepare(model)

        if self.args.res_opt:
            try:
                self.optimizer_ll.load_state_dict(data['opt_ll'])
                self.optimizer_hf.load_state_dict(data['opt_hf'])
                self.scheduler_ll.load_state_dict(data['scheduler_ll'])
                self.scheduler_hf.load_state_dict(data['scheduler_hf'])
            except:
                print_log(f"No optimizer", self.is_main)
            try:
                self.cur_epoch = data['epoch'] + 1
                self.cur_step = data['step']
                print("Current epoch", self.cur_epoch)
            except:
                print_log(f"No record epoch", self.is_main)
            
        if self.is_main:
            self.ema.load_state_dict(data['ema'])


    def train(self):
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):

            print(f"Training : {epoch+1}")
            epoch_start_time = time.time()
            self.cur_epoch = epoch
            self.wavelet_dual_model.train()
            
            for i, batch in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
                with self.accelerator.autocast(self.wavelet_dual_model):
                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])

                    if self.cur_step == 0:
                        for name, param in self.wavelet_dual_model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)   
    
                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.wavelet_dual_model.parameters(), 1.0)
                
                self.optimizer_ll.step()
                self.optimizer_hf.step()
                
                self.optimizer_ll.zero_grad()
                self.optimizer_hf.zero_grad()
                
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler_ll.step()
                    self.scheduler_hf.step()
                
                lr_ll = self.optimizer_ll.param_groups[0]['lr']
                lr_hf = self.optimizer_hf.param_groups[0]['lr']
           
                log_dict = {
                    'lr_ll': lr_ll,
                    'lr_hf': lr_hf,
                    'total_loss': loss_dict['total_loss'].item(),
                    'll_loss': loss_dict['ll_loss_dict']['total_loss'].item(),
                    'hf_loss': loss_dict['hf_loss_dict']['total_loss'].item(),
                }

                self.accelerator.log(log_dict, step=self.cur_step)
             
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
               
                if i % 200 == 0:
                    logging.info(state_str+'::'+str(log_dict))
                self.ema.update()

                self.cur_step += 1
             
                if self.cur_step == 1:
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon_l, radar_recon_h = self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            print("Datashape: ",batch.shape)
                            print_log(f" ========= Sanity Check over ==========", self.is_main)
                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)

            if self.args.valid:
                if (epoch+1)%5==0 or (epoch)==0:
                    cur_csi = self.test_samples(self.cur_step, (epoch+1))
        
                    if self.args.valid_limit:
                        self.save()
                    else:
                        if cur_csi != None and cur_csi > self.max_csi:
                            self.save('best')
                            print("Best model saved")
                            self.best_step = self.cur_step
                            self.max_csi = cur_csi
                        self.save('last')
                        print_log(f"Valid Results: {cur_csi}, Best csi: {self.max_csi}, Best step: {self.best_step}", self.is_main)
                    print_log(f" ========= Finish one Epoch ==========", self.is_main)
            else:
                self.save()
                print_log(f" ========= Finish one Epoch ==========", self.is_main)
                
            epoch_time = time.time() - epoch_start_time
            print_log(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")
            
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        
    def _get_seq_data(self, batch):
        return batch[:, :self.args.frames_out + self.args.frames_in]
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
        B = frames_in.shape[0] 

        frames_in_l, frames_in_h = self.dwt(frames_in)
        frames_out_l, frames_out_h = self.dwt(frames_out)

        frames_in_l = normalize(self.args.dataset, frames_in_l)
        frames_out_l = normalize(self.args.dataset, frames_out_l)

        frames_in_l = frames_in_l.view(B, self.args.frames_in, self.args.img_channel_L, self.args.img_size, self.args.img_size)
        frames_out_l = frames_out_l.view(B, self.args.frames_out, self.args.img_channel_L, self.args.img_size, self.args.img_size)

        frames_in_h = frames_in_h.view(B, self.args.frames_in, self.args.img_channel_H, self.args.img_size, self.args.img_size)
        frames_out_h = frames_out_h.view(B, self.args.frames_out, self.args.img_channel_H, self.args.img_size, self.args.img_size)

        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        
        if hasattr(self.wavelet_dual_model, 'module'):
            _, loss = self.wavelet_dual_model.module.predict(
                frames_in_ll=frames_in_l, 
                frames_in_hf=frames_in_h,
                frames_gt_ll=frames_out_l, 
                frames_gt_hf=frames_out_h,
                compute_loss=True
            )
        else:
            _, loss = self.wavelet_dual_model.predict(
                frames_in_ll=frames_in_l, 
                frames_in_hf=frames_in_h,
                frames_gt_ll=frames_out_l, 
                frames_gt_hf=frames_out_h,
                compute_loss=True
            )

        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        
        if isinstance(loss, dict):
            if 'total_loss' in loss:
                return loss
            else:
                raise ValueError("The loss must contain the 'total_loss' key.")
        else:
            return {'total_loss': loss}
        
    
    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False, vis_diff=False):
        if use_ema:
            sample_fn = self.ema.ema_model.module.predict if hasattr(self.ema.ema_model, 'module') else self.ema.ema_model.predict
        else:
            sample_fn = self.wavelet_dual_model.module.predict if hasattr(self.wavelet_dual_model, 'module') else self.wavelet_dual_model.predict
        
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]

        B = radar_input.shape[0]
        radar_input_l, radar_input_h = self.dwt(radar_input)
        radar_input_l = radar_input_l.view(B, self.args.frames_in, self.args.img_channel_L, self.args.img_size, self.args.img_size)
        
        radar_gt_l, radar_gt_h = self.dwt(radar_gt)
        radar_gt_l = radar_gt_l.view(B, self.args.frames_out, self.args.img_channel_L, self.args.img_size, self.args.img_size)

        radar_input_l = normalize(self.args.dataset, radar_input_l)
        radar_gt_l = normalize(self.args.dataset, radar_gt_l)

        radar_input_h = radar_input_h.view(B, self.args.frames_in, self.args.img_channel_H, self.args.img_size, self.args.img_size)
        radar_gt_h = radar_gt_h.view(B, self.args.frames_out, self.args.img_channel_H, self.args.img_size, self.args.img_size)

        (radar_pred_l, radar_pred_h), *_ = sample_fn(
            radar_input_l, 
            radar_input_h, 
            radar_gt_l, 
            radar_gt_h, 
            compute_loss=False
        )
        
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred_l = self.accelerator.gather(radar_pred_l).detach().cpu().numpy()
        radar_pred_h = self.accelerator.gather(radar_pred_h).detach().cpu().numpy()
        
        return radar_gt, radar_pred_l, radar_pred_h 
    
    
    def test_samples(self, milestone, epoch=None, do_test=False):
        if do_test==False:
            print("Validation")
        if do_test==True:
            print("Testing")
            
        data_loader = self.test_loader if do_test else self.valid_loader
        self.wavelet_dual_model.eval()

        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)
        
        if do_test:
            from utils.metrics import Evaluator
            eval = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            )
        else:
            from utils.metrics_valid import Evaluator
            eval = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            )
            
        valid_nums = 0
        for batch in tqdm(data_loader, total=len(data_loader)):
            radar_ori, radar_recon_L, radar_recon_H = self._sample_batch(batch)
            radar_recon_L = unnormalize_LL(self.args.dataset, radar_recon_L)
            # Reshape for IDWT
            # radar_recon_L: (B, T, 1, 64, 64)
            # radar_recon_H: (B, T, 3, 64, 64)
            B, T = radar_recon_L.shape[:2]
            
            # Convert to torch tensors
            yl_flat = torch.from_numpy(radar_recon_L).reshape(B*T, 1, 64, 64).float()
            yh_flat = torch.from_numpy(radar_recon_H).reshape(B*T, 3, 64, 64).float()
            
            # Reshape yh to (B*T, 1, 3, 64, 64) for IDWT
            yh_reshaped = yh_flat.unsqueeze(1)  # (B*T, 1, 3, 64, 64)
            
            # Apply IDWT
            radar_recon_flat = self.idwt((yl_flat, [yh_reshaped]))

            radar_ori = radar_ori/255
            radar_recon_flat = radar_recon_flat / 255
            # Reshape back to (B, T, 1, 128, 128)
            radar_recon = radar_recon_flat.reshape(B, T, 1, 128, 128).cpu().numpy()
            
            
            if self.is_main:
                eval.evaluate(radar_ori, radar_recon)
                
            valid_nums += 1
            if not do_test and self.args.valid_limit and valid_nums >= self.args.vlnum:
                break
                
        if self.is_main:
            res = eval.done()
            prefix = "test" if do_test else "val"
            
            log_data = {f"{prefix}/{k}": v for k, v in res.items()}
            log_data[f"{prefix}/epoch"] = epoch 
            
            if do_test:
                print_log(f"Test Results: {res}")
            else:
                print_log(f"Valid Results: {res}")
            print_log("="*30)

            res["epoch"] = epoch
            self.accelerator.log(log_data, step=self.cur_step)

            if self.args.valid:
                return res['csi'] 
        else:
            return None

        
    def check_milestones(self, target_ckpt=None):
        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples(saved_dir_name, do_test=True)
            print("Testing done")
            return
        
        mils_paths = os.listdir(self.ckpt_path)
        milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)

        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples(milestones[m], do_test=True)


def main():
    args = create_parser()
    exp = Runner(args)

    if args.gpu_use:
        gpu_list = ','.join(args.gpu_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    if not args.eval:
        exp.train()
    else:
        exp.check_milestones(target_ckpt=args.ckpt_milestone)
    

if __name__ == '__main__':
    main()
