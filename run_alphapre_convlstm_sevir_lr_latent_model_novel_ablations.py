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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm, BoundaryNorm
import matplotlib.colors as mcolors
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import ProjectConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs
from ema_pytorch import EMA
from diffusers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
from datasets.dataset_mosdac import *
from datasets.get_datasets import get_dataset
from utils.tools import print_log, cycle, show_img_info
from copy import deepcopy
from models.autoencoder_kl import AutoencoderKL
import importlib
# ========================================================
torch.backends.cudnn.deterministic = True
# ========================================================
# Apply your own wandb api key to log online
os.environ["WANDB_API_KEY"] = "6427ba1f8d0c13065720163c3aed0fa974031bef"
# os.environ["WANDB_SILENT"] = "true"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"


MODEL_REGISTRY = {
    # -------------------------------------------------------------------------
    # Existing Latent Space Models
    # -------------------------------------------------------------------------
    "alphapre": {
        "module": "models.Latent_space_models.alphapre_latent",
        "kwargs_type": "basic",
    },
    "alphapre_latent": {
        "module": "models.Latent_space_models.alphapre_latent",
        "kwargs_type": "basic",
    },
    "alphapre_latent_amplinet_mseonly": {
        "module": "models.Latent_space_models.alphapre_amplinet_MSE_only_latent",
        "kwargs_type": "basic",
    },
    "amplinet": {
        "module": "models.Full_space_models.alphapre_amplinet",
        "kwargs_type": "basic",
    },
    "amplinet_latent_falfcl": {
        "module": "models.Latent_space_models.alpha_amplinet_latent_FAL_FCL",
        "kwargs_type": "standard",
    },
    "amplinet_latent_falfcl_mse_hybrid": {
        "module": "models.Latent_space_models.alpha_amplinet_latent_hybrid_falfcl_MSE",
        "kwargs_type": "hybrid",
    },
    "alpha_fnoamplinet_latent_falfcl_var1": {
        "module": "models.Latent_space_models.alphapre_fnoamplinet_falfcl_only_variant1_latent",
        "kwargs_type": "standard",
    },
    "alpha_fnoamplinet_latent_falfcl": {
        "module": "models.Latent_space_models.alphapre_fnoamplinet_falfcl_only_latent",
        "kwargs_type": "standard",
    },
    "alpha_afnoamplinet_latent_falfcl": {
        "module": "models.Latent_space_models.alphapre_AFNOamplinet_falfcl_only_latent",
        "kwargs_type": "standard",
    },
}

# Ablation model prefixes that need dot→underscore conversion
ABLATION_PREFIXES = [
    "amplinet_latent_falfcl_only_",
]


def get_model_config(backbone: str, work: str) -> dict:
    """
    Get model config from registry, with automatic handling of ablation models.
    
    For ablation models like 'amplinet_latent_falfcl_only_2.3.3':
    - Converts dots to underscores for module path
    - Maps to: models.Model_parts_importance_latent_space.alpha_amplinet_latent_FAL_FCL_2_3_3
    """
    # Check if it's in the static registry
    if backbone in MODEL_REGISTRY:
        return MODEL_REGISTRY[backbone]
    
    # Check if it's an ablation model with dots
    for prefix in ABLATION_PREFIXES:
        if backbone.startswith(prefix):
            # Extract version part (e.g., "2.3.3" from "amplinet_latent_falfcl_only_2.3.3")
            version = backbone[len(prefix):]
            
            # Convert dots to underscores for the module name
            version_underscore = version.replace(".", "_")
            if version_underscore.split("_")[-1] == "hybridloss":
                kwargs_type = "hybrid"

            if version_underscore.split("_")[-1] == "final":
                kwargs_type = "gabor_convparallel_wavelet_final"

            elif "waveletgfngabor" in version_underscore.split("_")[-1]:
                kwargs_type = "gabor_gfn_wavelet"

            elif "convparallelwavelet" in version_underscore.split("_")[-1]:
                kwargs_type = "gabor_convparallel_wavelet"

            elif "mlpwavelets" in version_underscore.split("_")[-1]:
                kwargs_type = "mlp_wavelet"

            elif "waveletafnogabor" in version_underscore.split("_")[-1]:
                kwargs_type = "gabor_afno_wavelet"

            elif "groupedconvwaveletsgabor" in version_underscore.split("_")[-1]:
                kwargs_type = "gc_gabor_wavelet"

            elif "convwavelets" in version_underscore.split("_")[-1]:
                kwargs_type = "conv_wavelet"

            elif "waveletsgabor" in version_underscore.split("_")[-1] or "gaborconvwavelets" in version_underscore.split("_")[-1]:
                kwargs_type = "gabor_wavelet"

            elif "gaborhybrid" in version_underscore.split("_")[-1]:
                kwargs_type = "gaborhybrid"

            elif "afnogabor" in version_underscore.split("_")[-1]:
                kwargs_type = "afno_gabor_standared"
                
            elif "spectralgabor" in version_underscore.split("_")[-1]:
                kwargs_type = "spectralgabor"

            elif "gabor" in version_underscore.split("_")[-1]:
                kwargs_type = "gabor_standared"

            else:
                kwargs_type = "standared"

       
            # Build module path
            if work == "ablation":
                module_path = f"models.Ablations_final.alpha_amplinet_latent_FAL_FCL_{version_underscore}"

            elif work == "incremental":
                module_path = f"models.Incremental_model.alpha_amplinet_latent_FAL_FCL_{version_underscore}"

            return {
                "module": module_path,
                "kwargs_type": kwargs_type,
            }
    
    # Not found
    return None

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone',       type=str,   default='alpha_afnoamplinet_latent_falfcl',        help='backbone model for deterministic prediction (alphapre/convlstm_paper/simvp)')
    parser.add_argument("--seed",           type=int,   default=0,                 help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='meteo_lr_latent_32',      help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default='Testing_Integrity_with_afno_amplinet_0.01_1.0',              help="additional note for experiment")

    # --------------- Loss weights ---------------
    parser.add_argument("--mse_weight", type=float, default=0.00,            help="mse weight for hybid falfcl loss")
    parser.add_argument("--falfcl_weight", type=float, default=1.00,            help="falfcl weight for hybid falfcl loss")

    # ---------------------- Spectral --------------------
    parser.add_argument("--modes"           , type=int  , default=8,                help="modes for spectral")
    parser.add_argument("--afno_blocks"      , type=int  , default=1,               help="Number of blocks in afno")
    parser.add_argument("--afno2D_hidden_size_factor", type=int, default=1,         help="hidden size factor in afno2d")
    parser.add_argument("--afno_sparsity_threshold",   type=float, default=0.01,    help="sparsity threshold in afno2d")
    
    # ---------------------- ConvParallel Args --------------------
    parser.add_argument("--conv_kernel",    type=int  , default=3,              help="Conv parallel kernel value")
    parser.add_argument("--norm_before",    type=str2bool  , default=False,              help="want to use the norm before in convparallel")
    parser.add_argument("--use_residual",    type=str2bool  , default=False,              help="want to use the residual in convparallel setting")
    parser.add_argument("--adaptive_fusion",    type=str2bool  , default=False,              help="want to use the adaptive_fusion in convparallel setting")
    parser.add_argument("--channel_mixing",    type=str2bool  , default=False,              help="want to use the adaptive_fusion in convparallel setting")

    # ---------------------- Hidden dim --------------------
    parser.add_argument("--hidden_dim",    type=int  , default=64,              help="Hidden dimension inside the model")

    #----------------------- MLP Parameters---------------------
    parser.add_argument("--size_factor",    type=float  , default=1.0,              help="Hidden size factor for MLP")

    #----------------------- GFN ---------------------
    parser.add_argument("--num_gfn_layers",    type=int  , default=1,              help="Hidden size factor for MLP")

    #-----------------------Ablation Motive-------------------
    parser.add_argument("--work"       , type=str    ,  default='ablation',   help="incremental, ablation")
    
    #----------------------- Model Specific---------------------
    parser.add_argument("--residual_mode"       , type=str    ,  default='gabor',   help="residual connection to use in the model, values can be ['gabor', 'mlp', 'none']")
    parser.add_argument("--st_conv_groups"      , type=int    ,  default=1,         help="No. of groups in spatiotemporal conv")

    # --------------- Gabor Parameters ---------------
    parser.add_argument("--weight_scale_low"    , type=float, default=0.00,            help="weight_scale for gabor for low freq")
    parser.add_argument("--alpha_low"           , type=float, default=0.00,            help="alpha for gabor for low freq")
    parser.add_argument("--beta_low"            , type=float, default=0.00,            help="beta for gabor for low freq")
    parser.add_argument("--freq_multiplier_low" , type=float, default=0.00,            help="freq_multiplier for gabor for low freq")

    parser.add_argument("--weight_scale_high"    , type=float, default=0.00,            help="weight_scale for gabor for high freq")
    parser.add_argument("--alpha_high"           , type=float, default=0.00,            help="alpha for gabor for high freq")
    parser.add_argument("--beta_high"            , type=float, default=0.00,            help="beta for gabor for high freq")
    parser.add_argument("--freq_multiplier_high" , type=float, default=0.00,            help="freq_multiplier for gabor for high freq")

    parser.add_argument("--weight_scale"    , type=float, default=0.00,            help="weight_scale for gabor")
    parser.add_argument("--alpha"           , type=float, default=0.00,            help="alpha for gabor")
    parser.add_argument("--beta"            , type=float, default=0.00,            help="beta for gabor")
    parser.add_argument("--freq_multiplier" , type=float, default=0.00,            help="freq_multiplier for gabor")

    #------------------- Wavelet ----------------------
    parser.add_argument("--wave",      type=str,    default='haar',           help="Type of wavelet transform")
    parser.add_argument("--wavelet_level",     type=int,   default=1,         help="Wavelet level used for wavelet transform")
    parser.add_argument("--hf_mode",        type=str,    default= 'seperate',     help= "High frequency  mode" )

    # --------------- Dataset ---------------
    parser.add_argument("--dataset",            type=str,       default='meteo_lr_latent_32',   help="dataset name")
    parser.add_argument("--datatype",           type=str,       default='vil_vip',           help="Indicates the datatype available")
    parser.add_argument("--file_rain_seq_add",  type=str,       default=0,              help="Rainy days file")
    parser.add_argument("--method",             type= int,      default= None,          help = "Method to select the dataset as per the need. (Look at the function for more details)")
    parser.add_argument("--img_size",           type=int,       default=32,            help="image size")
    parser.add_argument("--stride",             type=int,       default=13,             help="dataset stride")
    parser.add_argument("--img_channel",        type=int,       default=4,              help="channel of image")
    parser.add_argument("--patch",              type=int,       default=2,              help="patch size")
    parser.add_argument("--seq_len",            type=int,       default=25,             help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",          type=int,       default=5,              help="nuFmber of frames to input")
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
    parser.add_argument("--warmup_steps",   type=int,   default=1000,             help="warmup steps")
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
    parser.add_argument("--vlnum",          type=int,   default=10,              help="valid limit nums")
    parser.add_argument("--visual",         action="store_true",                 help="save all test sample visualization")
    parser.add_argument("--gpu_use",        type=str,   nargs='+', default=["0",],  help="gpu(s) to use")
    parser.add_argument("--res_opt",        action="store_true",                 help="resume opt")  # Remember to activate this when you want to resume

    # --------------- Wandb ---------------
    parser.add_argument("--wandb_state",    type=str,   default='online',      help="wandb state config")
    parser.add_argument("--wandb_project_name", type=str, default="Alphapre", help="wandb project name")
    parser.add_argument("--run_name",       type=str,   default='Afno_Amplinet_falfcl_only_meteonet_latent_32_.21',        help="wandb run name")

    #------------------------- Plots -----------------------------
    parser.add_argument("--generate_outputs", action="store_true",               help="Generate visualizations from checkpoint")
    parser.add_argument("--plot_saving_directory", type=str,  default=None,      help="Enter saving directory for plots")

    #------------------------- AE --------------------------------
    parser.add_argument("--ae_ckpt_path", type=str, default="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints/autoencoder_checkpoint_32_METEONET.pth", help="ae ckpt path")
    args = parser.parse_args()
    return args

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
                # 'resume': self.args.ckpt_milestone
                }
                         }   # disabled, online, offline
        )
        print_log(f"Using GPUs: {self.device}", self.is_main)
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
    
        print_log(self.accelerator.state, self.is_main)
        
        self._load_data()
        self.train_loader, self.valid_loader, self.test_loader, self.valid_os_loader, self.test_os_loader = self.accelerator.prepare(
        self.train_loader, self.valid_loader, self.test_loader, self.valid_os_loader, self.test_os_loader)
        self._build_model()
        self._build_optimizer()
        # distributed ema for parallel sampling

        self.model, self.optimizer,  self.scheduler = self.accelerator.prepare(
            self.model, 
            self.optimizer, self.scheduler
        )
        
        self.train_dl_cycle = cycle(self.train_loader)
        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)
            # print_log(show_img_info(sample), self.is_main)
            
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        print_log(f"Input shape: {self.args.img_size}x{self.args.img_size}")


        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return self.accelerator.device
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = self.args.backbone
        self.exp_name   = f"{self.model_name}_{self.args.dataset}_{self.args.exp_note}"
        
        cur_dir         = os.path.dirname(os.path.abspath(__file__))
        
        self.exp_dir    = osp.join(cur_dir, 'Exps', self.args.exp_dir, self.exp_name)        
        self.ckpt_path  = osp.join(self.exp_dir, 'checkpoints')
        self.valid_path = osp.join(self.exp_dir, 'valid_samples')
        self.test_path  = osp.join(self.exp_dir, 'test_samples')
        self.log_path   = osp.join(self.exp_dir, 'logs')
        self.sanity_path = osp.join(self.exp_dir, 'sanity_check')
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        os.makedirs(self.valid_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

    #=================Automatically generating ckpt milestone path====================
        if self.args.eval and self.args.ckpt_milestone is None:
            self.args.ckpt_milestone = osp.join(self.ckpt_path, "ckpt-best.pt")
            print(f"[Auto] Using checkpoint: {self.args.ckpt_milestone}")   
    #=================================================================================

        exp_params      = self.args.__dict__
        params_path     = osp.join(self.exp_dir, 'params.yaml')
        yaml.dump(exp_params, open(params_path, 'w'))
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            # filemode='a',
            handlers=[
                logging.FileHandler(osp.join(self.log_path, 'log.log')),
                # logging.StreamHandler()
            ]
        )
    
    def load_autoencoder(
        self,
        model,
        checkpoint_path,
        device="cuda",
        dtype=torch.float32
    ):
        """
        model: instantiated autoencoder model (same architecture as training)
        checkpoint_path: path to .pt / .pth checkpoint
        """

        # ---- load checkpoint to CPU first (safe) ----
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        assert "model" in ckpt, "Checkpoint does not contain 'model' key"

        ckpt_model = ckpt["model"]
        
        # ---- find matching submodel key ----
        model_keys = list(model.state_dict().keys())
        
        ckpt_keys = list(ckpt_model.keys())
        
        # If checkpoint saved multiple submodels, pick autoencoder
        if isinstance(ckpt_model, dict) and all(isinstance(v, dict) for v in ckpt_model.values()):
            # typical structure: ckpt['model']['autoencoder_kl']
            if len(ckpt_model) == 1:
                ckpt_state = list(ckpt_model.values())[0]
            else:
                # explicitly choose autoencoder
                
                ckpt_state = ckpt_model.get("autoencoder_kl", None)
                if ckpt_state is None:
                    raise KeyError("autoencoder_kl not found in checkpoint")
                
                    
        else:
            ckpt_state = ckpt_model

        # ---- strip 'module.' if present ----
        new_state_dict = OrderedDict()
        for k, v in ckpt_state.items():
            if k.startswith("module."):
                k = k[7:]
            elif k.startswith("net."):
                k = k[4:]
            new_state_dict[k] = v

        
        # ---- load weights ----
        model.load_state_dict(new_state_dict, strict=True)

        # ---- move to device and eval ----
        model.to(device=device, dtype=dtype)
        model.eval()

        # ---- freeze params (important for compression) ----
        for p in model.parameters():
            p.requires_grad = False

        print("✅ Autoencoder loaded successfully")
        return model

    @torch.no_grad()
    def encode_stage(self, model, x, scale_factor):
        z = model.encode(x)
        return z.sample() * scale_factor


    @torch.no_grad()
    def decode_stage(self,model, z, scale_factor):
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z)
        z = z.to(next(model.parameters()).device)
        z = z / scale_factor
        return model.decode(z)

    def _load_data(self):
        # =================================
        # Get Train/Valid/Test dataloader among datasets 
        # =================================

        train_data, valid_data , test_data , color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            # data_path=self.args.data_path,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            batch_size=self.args.batch_size,
            stride=self.args.stride,
            file_rain_seq_add=self.args.file_rain_seq_add,
            method = self.args.method,
            in_channels = self.args.frames_in,
            out_channels = self.args.frames_out,
            preprocess_type = self.args.preprocessing
        )
        if self.args.dataset == "sevir_lr_latent_32":
            data_name = "sevir"
        elif self.args.dataset == "shanghai_lr_latent_32":
            data_name = "shanghai"
        elif self.args.dataset == "meteo_lr_latent_32":
            data_name = "meteo"
        elif self.args.dataset == "cikm_latent_32":
            data_name = "cikm"

        _, valid_os_data, test_os_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=data_name,
            # data_path=self.args.data_path,
            img_size=128,
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
        self.thresholds      = THRESHOLDS
        self.scale_value     = PIXEL_SCALE
        
    
        if self.args.dataset == 'vil_mosdac' or self.args.dataset == 'vil' or self.args.dataset == 'mosdac':
        
            self.train_loader = create_loader(train_data, batch_size= self.args.batch_size, shuffle=True)
            self.valid_loader = create_loader(valid_data, batch_size= self.args.batch_size)
            self.test_loader = create_loader(test_data, batch_size= self.args.batch_size)

        if self.args.dataset == 'sevir_lr_latent_32' or self.args.dataset == 'sevir_lr_latent':
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_os_loader = valid_os_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_os_loader = test_os_data.get_torch_dataloader(num_workers=self.args.num_workers)
        
        if self.args.dataset == 'shanghai_lr_latent_32' or self.args.dataset == 'meteo_lr_latent_32' or self.args.dataset == 'cikm_latent_32':
            self.train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=True
            )
            self.valid_loader = torch.utils.data.DataLoader(
                valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
            self.valid_os_loader = torch.utils.data.DataLoader(
                valid_os_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
            )
            self.test_os_loader = torch.utils.data.DataLoader(
                test_os_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
            )
        

        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",  # Returns the number of batches.
                  self.is_main)
        
        for sample in self.train_loader:
            print(sample.shape)
            break
        
        
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}",
                  self.is_main)
    
    
    def _build_model(self):
        """
        Build model using registry-based dynamic loading.
        Supports dots in backbone names (e.g., amplinet_latent_falfcl_only_2.3.3)
        which map to underscore files (e.g., alpha_amplinet_latent_FAL_FCL_2_3_3.py)
        """
        print_log("Build Model!", self.is_main)
        
        # Build autoencoder
        self.ae_model = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            down_block_types=('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'),
            up_block_types=('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'),
            block_out_channels=(128, 256, 512),
            layers_per_block=2,
            latent_channels=4,
            norm_num_groups=32
        )
        
        backbone = self.args.backbone
        
        # Get config (handles dot→underscore conversion for ablation models)
        config = get_model_config(backbone, self.args.work)
        
        if config is None:
            # List available options
            static_models = list(MODEL_REGISTRY.keys())
            ablation_examples = [
                "amplinet_latent_falfcl_only_1",
                "amplinet_latent_falfcl_only_2.1",
                "amplinet_latent_falfcl_only_2.3.1",
                "amplinet_latent_falfcl_only_2.3.2.1",
                "amplinet_latent_falfcl_only_3.1",
                "etc..."
            ]
            available = '\n  - '.join(static_models + ablation_examples)
            raise NotImplementedError(
                f"Backbone '{backbone}' not found.\nAvailable backbones:\n  - {available}"
            )
        
        module_path = config["module"]
        kwargs_type = config["kwargs_type"]
        
        # Dynamic import
        print_log(f"Loading module: {module_path}", self.is_main)
        module = importlib.import_module(module_path)
        get_model = module.get_model
        
        # Calculate total_steps
        total_steps = self.args.epochs * len(self.train_loader)
        
       
        # Build kwargs
        if kwargs_type == "basic":
            kwargs = {
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "n_layers": self.args.layers,
                "pha_weight": self.args.pha_weight,
                "anet_weight": self.args.anet_weight,
                "amp_weight": self.args.amp_weight,
                "spec_num": self.args.spec_num,
                "aweight_stop_steps": self.args.aw_stop_step,
            }

        elif kwargs_type == "standared":
            kwargs = {
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "n_layers": self.args.layers,
                "pha_weight": self.args.pha_weight,
                "anet_weight": self.args.anet_weight,
                "amp_weight": self.args.amp_weight,
                "spec_num": self.args.spec_num,
                "aweight_stop_steps": self.args.aw_stop_step,
            }

        elif kwargs_type == "gabor_standared":
            kwargs = {
                "weight_scale": self.args.weight_scale,
                "alpha": self.args.alpha,
                "beta": self.args.beta,
                "freq_multiplier": self.args.freq_multiplier,
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "n_layers": self.args.layers,
                "pha_weight": self.args.pha_weight,
                "anet_weight": self.args.anet_weight,
                "amp_weight": self.args.amp_weight,
                "spec_num": self.args.spec_num,
                "aweight_stop_steps": self.args.aw_stop_step,
            }

        elif kwargs_type == "gc_gabor_wavelet":
            kwargs = {
                "weight_scale_low": self.args.weight_scale_low,
                "alpha_low": self.args.alpha_low,
                "beta_low": self.args.beta_low,
                "freq_multiplier_low": self.args.freq_multiplier_low,
                "weight_scale_high": self.args.weight_scale_high,
                "alpha_high": self.args.alpha_high,
                "beta_high": self.args.beta_high,
                "freq_multiplier_high": self.args.freq_multiplier_high,
                "wave": self.args.wave, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "hf_mode" : self.args.hf_mode,
                "dim": 64, 
                "st_conv_groups": self.args.st_conv_groups
            }

        elif kwargs_type == "gabor_wavelet":
            kwargs = {
                "weight_scale_low": self.args.weight_scale_low,
                "alpha_low": self.args.alpha_low,
                "beta_low": self.args.beta_low,
                "freq_multiplier_low": self.args.freq_multiplier_low,
                "weight_scale_high": self.args.weight_scale_high,
                "alpha_high": self.args.alpha_high,
                "beta_high": self.args.beta_high,
                "freq_multiplier_high": self.args.freq_multiplier_high,
                "wave": self.args.wave, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "hf_mode" : self.args.hf_mode,
                "dim": 64
            }

        elif kwargs_type == "gabor_gfn_wavelet":
            kwargs = {
                "num_gfn_layers": self.args.num_gfn_layers, 
                "weight_scale_low": self.args.weight_scale_low,
                "alpha_low": self.args.alpha_low,
                "beta_low": self.args.beta_low,
                "freq_multiplier_low": self.args.freq_multiplier_low,
                "weight_scale_high": self.args.weight_scale_high,
                "alpha_high": self.args.alpha_high,
                "beta_high": self.args.beta_high,
                "freq_multiplier_high": self.args.freq_multiplier_high,
                "wave": self.args.wave, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "hf_mode" : self.args.hf_mode,
                "dim": 128
            }

        elif kwargs_type == "gabor_convparallel_wavelet":
            kwargs = {
                "weight_scale_low": self.args.weight_scale_low,
                "alpha_low": self.args.alpha_low,
                "beta_low": self.args.beta_low,
                "freq_multiplier_low": self.args.freq_multiplier_low,
                "weight_scale_high": self.args.weight_scale_high,
                "alpha_high": self.args.alpha_high,
                "beta_high": self.args.beta_high,
                "freq_multiplier_high": self.args.freq_multiplier_high,
                "wave": self.args.wave, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "hf_mode" : self.args.hf_mode,
                "dim": 64, 
                "afno_blocks": self.args.afno_blocks, 
                "sparsity_threshold": self.args.afno_sparsity_threshold, 
                "afno_hidden_size_factor": self.args.afno2D_hidden_size_factor,
                "k_spatial": self.args.conv_kernel, 
                "norm_before": self.args.norm_before, 
                "if_residual": self.args.use_residual, 
                "adapt_fusion": self.args.adaptive_fusion, 
                "channel_mixing": self.args.channel_mixing
            }

        elif kwargs_type == "gabor_convparallel_wavelet_final":
            kwargs = {
                "weight_scale_low": self.args.weight_scale_low,
                "alpha_low": self.args.alpha_low,
                "beta_low": self.args.beta_low,
                "freq_multiplier_low": self.args.freq_multiplier_low,
                "weight_scale_high": self.args.weight_scale_high,
                "alpha_high": self.args.alpha_high,
                "beta_high": self.args.beta_high,
                "freq_multiplier_high": self.args.freq_multiplier_high,
                "wave": self.args.wave, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "hf_mode" : self.args.hf_mode,
                "dim": self.args.hidden_dim, 
                "afno_blocks": self.args.afno_blocks, 
                "sparsity_threshold": self.args.afno_sparsity_threshold, 
                "afno_hidden_size_factor": self.args.afno2D_hidden_size_factor,
                "k_spatial": self.args.conv_kernel,
            }

        elif kwargs_type == "gabor_afno_wavelet":
            kwargs = {
                "weight_scale_low": self.args.weight_scale_low,
                "alpha_low": self.args.alpha_low,
                "beta_low": self.args.beta_low,
                "freq_multiplier_low": self.args.freq_multiplier_low,
                "weight_scale_high": self.args.weight_scale_high,
                "alpha_high": self.args.alpha_high,
                "beta_high": self.args.beta_high,
                "freq_multiplier_high": self.args.freq_multiplier_high,
                "wave": self.args.wave, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "hf_mode" : self.args.hf_mode,
                "dim": 64, 
                "afno_blocks": self.args.afno_blocks, 
                "sparsity_threshold": self.args.afno_sparsity_threshold, 
                "afno_hidden_size_factor": self.args.afno2D_hidden_size_factor
            }

        elif kwargs_type == "conv_wavelet":
            kwargs = {
                "weight_scale": self.args.weight_scale,
                "alpha": self.args.alpha,
                "beta": self.args.beta,
                "freq_multiplier": self.args.freq_multiplier,
                "wave": self.args.wave, 
                "size_factor": self.args.size_factor, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "residual_mode": self.args.residual_mode, 
                "dim": 64,
            }

        elif kwargs_type == "afno_gabor_standared":
            kwargs = {
                "weight_scale": self.args.weight_scale,
                "alpha": self.args.alpha,
                "beta": self.args.beta,
                "freq_multiplier": self.args.freq_multiplier,
                "afno_blocks": self.args.afno_blocks, 
                "afno2D_hidden_size_factor": self.args.afno2D_hidden_size_factor, 
                "afno_sparsity_threshold": self.args.afno_sparsity_threshold,
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "n_layers": self.args.layers,
                "pha_weight": self.args.pha_weight,
                "anet_weight": self.args.anet_weight,
                "amp_weight": self.args.amp_weight,
                "spec_num": self.args.spec_num,
                "aweight_stop_steps": self.args.aw_stop_step,
            }

        elif kwargs_type == "gaborhybrid":
            kwargs = {
                "lambda1": self.args.mse_weight,
                "lambda2": self.args.falfcl_weight,
                "weight_scale": self.args.weight_scale,
                "alpha": self.args.alpha,
                "beta": self.args.beta,
                "freq_multiplier": self.args.freq_multiplier,
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "n_layers": self.args.layers,
                "pha_weight": self.args.pha_weight,
                "anet_weight": self.args.anet_weight,
                "amp_weight": self.args.amp_weight,
                "spec_num": self.args.spec_num,
                "aweight_stop_steps": self.args.aw_stop_step,
            }
        
        elif kwargs_type == "spectralgabor":
            kwargs = {
                "modes": self.args.modes,
                "weight_scale": self.args.weight_scale,
                "alpha": self.args.alpha,
                "beta": self.args.beta,
                "freq_multiplier": self.args.freq_multiplier,
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "n_layers": self.args.layers,
                "pha_weight": self.args.pha_weight,
                "anet_weight": self.args.anet_weight,
                "amp_weight": self.args.amp_weight,
                "spec_num": self.args.spec_num,
                "aweight_stop_steps": self.args.aw_stop_step,
            }

        elif kwargs_type == "hybrid":
            kwargs = {
                "lam1": self.args.mse_weight,
                "lam2": self.args.falfcl_weight,
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": 64,
                "n_layers": self.args.layers,
                "pha_weight": self.args.pha_weight,
                "anet_weight": self.args.anet_weight,
                "amp_weight": self.args.amp_weight,
                "spec_num": self.args.spec_num,
                "aweight_stop_steps": self.args.aw_stop_step,
            }
        
        elif kwargs_type == "mlp_wavelet":
            kwargs = {
                "size_factor": self.args.size_factor, 
                "wave": self.args.wave, 
                "wavelet_level": self.args.wavelet_level, 
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "hf_mode" : self.args.hf_mode,
                "dim": 64
            }

        # Create model
        model = get_model(**kwargs)
        
        print_log(f"model parameters : {kwargs}", self.is_main)
        
        self.model = model
        print_log("begin ema", self.is_main)
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)
        print_log("end device", self.is_main)
        
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total / 1e6), self.is_main)
            self.model_params = total

    def _build_optimizer(self):
        # =================================
        # Calcutate training nums and config optimizer and learning schedule
        # =================================
        num_steps_per_epoch = len(self.train_loader)
        # num_epoch = math.ceil(self.args.training_steps / num_steps_per_epoch)
        
        # self.global_epochs = max(num_epoch, self.args.epochs)
        self.global_epochs = self.args.epochs
        self.global_steps = self.global_epochs * num_steps_per_epoch
        self.steps_per_epoch = num_steps_per_epoch
        
        self.cur_step, self.cur_epoch = 0, 0

        warmup_steps = self.args.warmup_steps

        # Schedulers takes from diffusers.
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
       
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            betas=(self.args.lr_beta1, self.args.lr_beta2),
            weight_decay=self.args.l2_norm
        )
        if self.args.scheduler == 'constant':
            self.scheduler = get_constant_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
            )
        elif self.args.scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps, 
                num_training_steps=self.global_steps,
            )
        elif self.args.scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=warmup_steps , 
                num_training_steps=self.global_steps,
            )
        else:train
    
    def save(self, svname=None):
        # =================================
        # Save checkpoint state for model and ema
        # =================================
        if not self.is_main:
            return
        
        data = {
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        
        if svname == None:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt"))
            print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        else:
            torch.save(data, osp.join(self.ckpt_path, f"ckpt-{svname}.pt"))
            print_log(f"Save {svname} checkpoint to {self.ckpt_path}", self.is_main)

        
    def load(self, milestone):
        # =================================
        # load model checkpoint
        # =================================        
        device = self.accelerator.device
        
        if isinstance(milestone, str) and '.pt' in milestone:
            data = torch.load(milestone, map_location=device)
            print_log(f"Load checkpoint {milestone}.", self.is_main)
        else:
            data = torch.load(osp.join(self.ckpt_path, f"ckpt-{milestone}.pt"), map_location=device)
            print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.model = self.accelerator.prepare(model)
        if self.args.res_opt:
            try:
                self.optimizer.load_state_dict(data['opt'])
                self.scheduler.load_state_dict(data['scheduler'])
            except:
                print_log(f"No optimizer", self.is_main)
            try:
                self.cur_epoch = data['epoch'] + 1 
            except:
                print_log(f"No record epoch", self.is_main)

            try:
                self.cur_step = data['step']
            except:
                print_log(f"No record step", self.is_main)
            
        if self.is_main:
            ema_dict = data['ema']
            for key, value in ema_dict.items():
                # If the checkpoint has a scalar (size []), but we need a vector (size [1])
                if value.dim() == 0:
                    ema_dict[key] = value.unsqueeze(0)

            # 3. Load the fixed dictionary
            self.ema.load_state_dict(ema_dict)


    def train(self):
        # set global step as traing process
        # torch.autograd.set_detect_anomaly(True)
        self.ae = self.load_autoencoder(self.ae_model, self.ae_ckpt, "cuda")
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):

            print(f"Training : {epoch+1}")
            epoch_start_time = time.time()
            self.cur_epoch = epoch
            self.model.train()
            
            for i, batch in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
                # train the model with mixed_precision
                with self.accelerator.autocast(self.model):
               
                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])
                    
                    if self.cur_step == 0:
                        # training process check
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)   
    
                self.accelerator.wait_for_everyone()
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()
                
                # record train info
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = dict()
                log_dict['lr'] = lr
                for k,v in loss_dict.items():
                    if type(v) == float:
                        log_dict[k] = v
                    else:
                        log_dict[k] = v.item()
          
                self.accelerator.log(log_dict, step=self.cur_step)
             
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
               
            
                # update ema param and log file every 20 steps
                if i % 200 == 0:
                    logging.info(state_str+'::'+str(log_dict))
                self.ema.update()

                self.cur_step += 1
             
                
                # do santy check at begining
                if self.cur_step == 1:
                    """ santy check """
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon= self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            # if self.is_main:
                            #     for i in range(radar_ori.shape[0]):
                            #         self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(self.sanity_path, f"{i}/vil"),data_type='vil')
                            print_log(f" ========= Sanity Check over ==========", self.is_main)
                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)

            # save checkpoint and do test every epoch
            if self.args.valid:

                if (epoch+1)%5==0 or epoch==0:
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
                    print_log(f" ========= Finisth one Epoch ==========", self.is_main)
                    
            else:
                self.save()
                print_log(f" ========= Finisth one Epoch ==========", self.is_main)
            epoch_time = time.time() - epoch_start_time
            print_log(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")

            if (epoch+1)==30:
                self.accelerator.wait_for_everyone()
                self.accelerator.end_training()
                break

            time.sleep(10)
        
    def _get_seq_data(self, batch):
        # frame_seq = batch['vil'].unsqueeze(2).to(self.device)
        return batch[:, :self.args.frames_out + self.args.frames_in]       # [B, T, C, H, W]
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)    
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]
        std_val = frames_in.std()
    
        frames_in = frames_in/std_val
        frames_out = frames_out/std_val
        
        
        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        
        if hasattr(self.model, 'module'):
            _, loss = self.model.module.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
        else:
            _, loss = self.model.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        
        # ========================================Incase we want to plot gradient's influence ratio===============================================
        # falfcl_loss = loss['falfcl_loss']
        # mse_loss = loss['hf_loss']

        # falfcl_grads = torch.autograd.grad(falfcl_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        # falfcl_norm = torch.norm(
        #     torch.stack([torch.norm(g, 2) for g in falfcl_grads if g is not None]), 2
        # ).item()
        # mse_grads = torch.autograd.grad(mse_loss, self.model.parameters(), retain_graph=True, allow_unused=True)
        # mse_norm = torch.norm(
        #     torch.stack([torch.norm(g, 2) for g in mse_grads if g is not None]), 2
        # ).item()
        # inflence_ratio = falfcl_norm/mse_norm

        # loss['inflence_ratio'] = inflence_ratio
        #===========================================================================================================================================
        if isinstance(loss, dict):
            if 'total_loss' in loss:
                return loss
            else:
                raise ValueError("The loss must contain the 'total_loss' key.")
        else:
            return {'total_loss': loss}
        

    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False, vis_diff=False):
        # sample_fn = self.ema.ema_model.predict if use_ema else self.model.predict
        # First priority given to ema_model.
        sample_fn = (self.ema.ema_model.module.predict if hasattr(self.ema.ema_model, 'module') else self.ema.ema_model.predict) if use_ema else (self.model.module.predict if hasattr(self.model, 'module') else self.model.predict)
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        std_value = radar_input.std()
        radar_input = radar_input/std_value
        radar_pred, *_ = sample_fn(radar_input,compute_loss=False)
        radar_pred = radar_pred*std_value
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()
        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()

        return radar_gt, radar_pred
    
    
    def test_samples(self, milestone, epoch=None, do_test=False):
        if do_test==False:
            print("Validation")
        if do_test==True:
            print("Testing")
            self.ae = self.load_autoencoder(self.ae_model, self.ae_ckpt, "cuda")
       
        save_vis = True
        # init test data loader
        if do_test:
            data_loaders = zip(self.test_loader, self.test_os_loader)
        else:
            data_loaders = zip(self.valid_loader, self.valid_os_loader)

        # init sampling method
        self.model.eval()
        # init test dir config

        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)
        # if self.is_main:
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
            
        # start test loop
        valid_nums = 0
        assert len(self.test_loader) == len(self.test_os_loader), "Mismatch in lengths of test_loader and test_os_loader (might be due to batch size)"
        total = len(self.test_loader)
        for (batch, os_batch) in tqdm(data_loaders, total=total):
            
            radar_os_batch = self._get_seq_data(os_batch)
            radar_os_gt = radar_os_batch[:,self.args.frames_in:]

            _, radar_recon = self._sample_batch(batch)
            B, T, C, H, W = radar_recon.shape

            # flatten time
            radar_recon_flat = radar_recon.reshape(B*T, C, H, W)

            # decode once
            radar_recon_dec = self.decode_stage(self.ae, radar_recon_flat, 1.0)

            # reshape back
            radar_recon = radar_recon_dec.view(B, T, 1, 128, 128)


            radar_ori = radar_os_gt.cpu().numpy()
            radar_recon = radar_recon.cpu().numpy()

            
            # evaluate result and save
            if self.is_main:
                eval.evaluate(radar_ori, radar_recon)


            self.accelerator.wait_for_everyone()
            valid_nums += 1
            if not do_test and self.args.valid_limit and valid_nums >= self.args.vlnum:
                break
        # test done
        if self.is_main:
            
            res = eval.done()
            if self.is_main and self.args.eval:
                from utils.results_logger_csv import ResultsLogger
                logger = ResultsLogger(csv_path="/home/vatsal/Dataserver2/Neurips/Final_model_ablations.csv")
                logger.log_results(
                    res_dict=res,
                    backbone=self.args.backbone,
                    exp_note=self.args.exp_note,
                    dataset=self.args.dataset,
                    model_params=self.model_params,
                )

            prefix = "test" if do_test else "val"
            
            # Create a new dictionary with prefixed keys (e.g., 'val/csi', 'test/mse')
            log_data = {f"{prefix}/{k}": v for k, v in res.items()}
            
            # Add epoch/step info if needed (WandB handles step automatically via the 'step' arg, 
            # but sometimes it's nice to have epoch as an explicit metric)
            log_data[f"{prefix}/epoch"] = epoch 
            
            if do_test:
                print_log(f"Test Results: {res}")
            else:
                print_log(f"Valid Results: {res}")
            print_log("="*30)

            # Log the PREFIXED data
            self.accelerator.log(log_data, step=self.cur_step)
        
            # --- END FIX ---

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
        
        # In case of multiple milestones.
        mils_paths = os.listdir(self.ckpt_path)
        milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)

        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples(milestones[m], do_test=True)
            break
    
    
    

def main():
    args = create_parser()
    exp = Runner(args)

    if args.gpu_use:
        gpu_list = ','.join(args.gpu_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    if args.generate_outputs:
        # When just evaluating and visualizing
        save_dir = args.plot_saving_directory
        exp.generate_outputs_from_checkpoint(save_dir, data_type = args.datatype, target_ckpt=args.ckpt_milestone)

    else: 
        if not args.eval:
            exp.train()
            # exp.check_milestones()
        else:
            
            exp.check_milestones(target_ckpt=args.ckpt_milestone)
    

if __name__ == '__main__':
    main()