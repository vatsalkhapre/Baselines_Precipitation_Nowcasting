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
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
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


def get_model_config(backbone: str) -> dict:
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
            elif "gaborhybrid" in version_underscore.split("_")[-1]:
                kwargs_type = "gaborhybrid"
            elif "supplimentrygabor" in version_underscore.split("_")[-1]:
                kwargs_type = "gabor_supplimentry_std"
                
            elif "gabor" in version_underscore.split("_")[-1]:
                kwargs_type = "gabor_standared"
            else:
                kwargs_type = "standared"

            # Build module path
            module_path = f"models.model_novelty.alpha_amplinet_latent_FAL_FCL_{version_underscore}"
            
            return {
                "module": module_path,
                "kwargs_type": kwargs_type,
            }
    
    # Not found
    return None

#===================================================================================================
#                                           PLOTTING CODE                                          #
#===================================================================================================

def plot_image_sequence_colored(images_colored, path_save_imgs, 
                                title_prefix="t", 
                                cmap=None, norm=None, label=None):
    """
    Plot a sequence of pre-colored RGBA images. Optionally add a colorbar
    if cmap and norm are provided.

    Args:
        images_colored : np.ndarray of shape (T, H, W, 4), RGBA float64 from gray2color.
        path_save_imgs : str, full path to save the figure.
        title_prefix   : str, prefix for subplot titles.
        cmap           : matplotlib colormap (optional, for colorbar).
        norm           : matplotlib norm (optional, for colorbar).
        label          : str, colorbar label.
    """
    T = images_colored.shape[0]

    if T <= 10:
        nrows, ncols = 1, T
        fig_w = 2 * T
        fig_h = 2.8
    else:
        nrows, ncols = 2, (T + 1) // 2
        fig_w = 2 * ncols
        fig_h = 5.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    if T == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    for i in range(T):
        axes_flat[i].imshow(images_colored[i], interpolation='nearest')
        axes_flat[i].axis("off")

    # Hide extra axes (if T is odd and nrows=2)
    for j in range(T, len(axes_flat)):
        axes_flat[j].axis("off")

    # Add colorbar if cmap and norm are provided
    if cmap is not None and norm is not None:
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes_flat[:T].tolist(),
            orientation='horizontal',
            fraction=0.05, pad=0.08, aspect=40
        )
        cbar.ax.tick_params(labelsize=7)
        if label:
            cbar.set_label(label, fontsize=9)

    fig.savefig(path_save_imgs, dpi=300, bbox_inches='tight')
    plt.close(fig)

def denorm_and_colorize(seq, pixel_scale, gray2color_fn, data_type='vil'):
    """
    Denormalize a sequence and apply the dataset-specific gray2color.

    Args:
        seq            : np.ndarray, shape (T, H, W), values in [0, 1] float.
        pixel_scale    : float (e.g. 255 for VIL).
        gray2color_fn  : callable, the gray2color function from the dataset.
        data_type      : str, passed to gray2color if needed.

    Returns:
        colored : np.ndarray, shape (T, H, W, 4), RGBA float64.
        denormed: np.ndarray, shape (T, H, W), denormalized uint8 values.
    """
    seq_clipped = np.clip(seq, 0, 1)
    denormed = (seq_clipped * pixel_scale).astype(np.uint8)
    colored = np.array(
        [gray2color_fn(denormed[i], data_type=data_type) for i in range(len(denormed))],
        dtype=np.float64
    )
    return colored


# Helper to resolve the plot directory from ckpt_milestone path
# =============================================================================

def resolve_plot_dir(ckpt_milestone_path):
    """
    Given ckpt_milestone (e.g. /path/to/Exps/<exp_dir>/<exp_name>/checkpoints/ckpt-best.pt)
    return /path/to/Exps/<exp_dir>/plots/

    Logic: parent of ckpt file -> 'checkpoints' dir -> parent is <exp_name>,
           one more parent -> <exp_dir>, create plots/ there.
    """
    if os.path.isfile(ckpt_milestone_path):
        ckpt_dir = os.path.dirname(ckpt_milestone_path)       # .../checkpoints/
    else:
        ckpt_dir = ckpt_milestone_path

    exp_name_dir = os.path.dirname(ckpt_dir)                   # .../<exp_name>/
         
    plot_dir     = os.path.join(exp_name_dir, "plots")
    return plot_dir


#Extract cmap and norm from gray2color for colorbar
# =============================================================================

def extract_cmap_norm_from_gray2color(gray2color_fn):
    """
    gray2color internally builds:
        cmap = colors.ListedColormap(COLOR_MAP)
        norm = colors.BoundaryNorm(BOUNDS, cmap.N)

    We replicate that here by calling gray2color's closure variables.

    If gray2color is a simple function that uses module-level COLOR_MAP and BOUNDS,
    you can import those directly:
        from <your_module> import COLOR_MAP, BOUNDS

    Otherwise, the safest approach is to reconstruct from the function's globals
    or __code__. But the EASIEST solution is just to import them.
    
    ---
    
    RECOMMENDED: In your gray2color module, expose COLOR_MAP and BOUNDS as module 
    attributes, then do:
    
        from <module> import COLOR_MAP, BOUNDS
        cmap = colors.ListedColormap(COLOR_MAP)
        norm = colors.BoundaryNorm(BOUNDS, cmap.N)
        return cmap, norm
    
    Below is a fallback that tries to extract from gray2color's globals:
    """
    try:
        # Try to get COLOR_MAP and BOUNDS from gray2color's global scope
        g = gray2color_fn.__globals__
        COLOR_MAP = g.get('COLOR_MAP')
        BOUNDS = g.get('BOUNDS')
        
        if COLOR_MAP is not None and BOUNDS is not None:
            cmap = colors.ListedColormap(COLOR_MAP)
            norm = colors.BoundaryNorm(BOUNDS, cmap.N)
            return cmap, norm
    except AttributeError:
        pass
    
    # Fallback: return None (colorbar won't be drawn)
    print("[Plot Warning] Could not extract cmap/norm from gray2color. "
          "Colorbar will be skipped. To fix, expose COLOR_MAP and BOUNDS "
          "as module-level variables in your gray2color module.")
    return None, None

def subsample_frames(seq, target_count=10):
    """
    Subsample `target_count` frames from a sequence using odd indices.
    
    For 20 frames -> indices [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    For 10 frames -> all frames returned as-is
    For  5 frames -> all frames returned as-is

    Args:
        seq : np.ndarray, shape (T, ...).
        target_count : int, desired number of frames.

    Returns:
        np.ndarray of shape (target_count, ...) or (T, ...) if T <= target_count.
    """
    T = seq.shape[0]
    if T <= target_count:
        return seq
    indices = list(range(1, T, 2))  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    return seq[indices]


def resolve_plot_dir(ckpt_milestone_path):
    """
    Given ckpt_milestone path, return <grandparent>/plots/.
    
    e.g. .../Exps/<exp_dir>/<exp_name>/checkpoints/ckpt-best.pt
         -> .../Exps/<exp_dir>/plots/
    """
    if os.path.isfile(ckpt_milestone_path):
        ckpt_dir = os.path.dirname(ckpt_milestone_path)       # .../checkpoints/
    else:
        ckpt_dir = ckpt_milestone_path

    exp_name_dir = os.path.dirname(ckpt_dir)                   # .../<exp_name>/
    exp_dir      = os.path.dirname(exp_name_dir)                # .../<exp_dir>/
    plot_dir     = os.path.join(exp_dir, "plots")
    return plot_dir

def str2bool(v):
    return v.lower() in ('true', '1', 'yes')

#===================================================================================================
#                                           Args                                       #
#===================================================================================================


def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone',       type=str,   default='alpha_afnoamplinet_latent_falfcl',        help='backbone model for deterministic prediction (alphapre/convlstm_paper/simvp)')
    parser.add_argument("--seed",           type=int,   default=0,                 help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,   default='',      help="experiment directory")
    parser.add_argument("--exp_note",       type=str,   default='',              help="additional note for experiment")

    # --------------- Loss weights ---------------
    parser.add_argument("--mse_weight", type=float, default=0.00,            help="mse weight for hybid falfcl loss")
    parser.add_argument("--falfcl_weight", type=float, default=1.00,            help="falfcl weight for hybid falfcl loss")

    # --------------- Plotting Arguments ---------------
    parser.add_argument("--plot",         action="store_true",           help="Enable plotting during testing")
    parser.add_argument("--plot_stride",  type=int,   default=4,        help="Plot every N-th test batch (offset/stride)")
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

    #----------------------- GFN ---------------------
    parser.add_argument("--num_gfn_layers",    type=int  , default=1,              help="Hidden size factor for MLP")
    
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
    parser.add_argument("--hf_mode",        type=str,    default= 'separate',     help= "High frequency  mode" )

    #-----------------Other Parameters----------------
    parser.add_argument("--size_factor",  type=float, default=1.0,            help="factor for hidden layer of mlp")
    parser.add_argument("--hidden_dim",     type=int,   default=64,             help="Conv Resnet block hidden dimension")
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
        self.gray2color_fn   = color_save_fn.keywords['gray2color']
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
        config = get_model_config(backbone)
        
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

        elif kwargs_type == "gabor_supplimentry_std":
            kwargs = {
                "weight_scale": self.args.weight_scale,
                "alpha": self.args.alpha,
                "beta": self.args.beta,
                "freq_multiplier": self.args.freq_multiplier,
                "size_factor": self.args.size_factor,
                "total_steps": total_steps,
                "const_ratio": 0.1,
                "input_shape": (self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "img_channels": self.args.img_channel,
                "dim": self.args.hidden_dim,
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
                    time.sleep(30)
            else:
                self.save()
                print_log(f" ========= Finisth one Epoch ==========", self.is_main)
            epoch_time = time.time() - epoch_start_time
            print_log(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        
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
    
    
    def test_samples_with_plotting(self, milestone, epoch=None, do_test=False):
        """Drop-in replacement for Runner.test_samples with optional plotting."""

        if do_test == False:
            print("Validation")
        if do_test == True:
            print("Testing")
            self.ae = self.load_autoencoder(self.ae_model, self.ae_ckpt, "cuda")

        # ---- plotting setup ----                                               # <<< PLOT
        do_plot = getattr(self.args, 'plot', False) and do_test                  # <<< PLOT
        if do_plot:                                                               # <<< PLOT
            if self.args.ckpt_milestone is not None:                              # <<< PLOT
                plot_base = resolve_plot_dir(self.args.ckpt_milestone)            # <<< PLOT
            else:                                                                 # <<< PLOT
                plot_base = osp.join(self.exp_dir, '..', 'plots')                # <<< PLOT
            plot_base = os.path.abspath(plot_base)                                # <<< PLOT
                                                                                # <<< PLOT
            input_dir  = osp.join(plot_base, "Input")                             # <<< PLOT
            gt_dir     = osp.join(plot_base, "Ground_truth")                      # <<< PLOT
            pred_dir   = osp.join(plot_base, "Predicted")                         # <<< PLOT
            os.makedirs(input_dir,  exist_ok=True)                                # <<< PLOT
            os.makedirs(gt_dir,     exist_ok=True)                                # <<< PLOT
            os.makedirs(pred_dir,   exist_ok=True)                                # <<< PLOT
                                                                                # <<< PLOT
            plot_stride = getattr(self.args, 'plot_stride', 4)                    # <<< PLOT
            print(f"[Plot] Saving plots to: {plot_base}")                         # <<< PLOT
            print(f"[Plot] Plot stride: every {plot_stride} batches")             # <<< PLOT

        save_vis = True
        # init test data loader
        if do_test:
            data_loaders = zip(self.test_loader, self.test_os_loader)
        else:
            data_loaders = zip(self.valid_loader, self.valid_os_loader)

        # init sampling method
        self.model.eval()
        # init test dir config
        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test \
                else osp.join(self.valid_path, f"sample-{milestone}")
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

        # start test loop
        valid_nums = 0
        assert len(self.test_loader) == len(self.test_os_loader), \
            "Mismatch in lengths of test_loader and test_os_loader (might be due to batch size)"
        total = len(self.test_loader)

        sample_counter = 0                                                        # <<< PLOT

        for batch_idx, (batch, os_batch) in enumerate(tqdm(data_loaders, total=total)):

            radar_os_batch = self._get_seq_data(os_batch)
            radar_os_input = radar_os_batch[:, :self.args.frames_in]              # <<< PLOT  (B, T_in, C, 128, 128)
            radar_os_gt    = radar_os_batch[:, self.args.frames_in:]

            _, radar_recon = self._sample_batch(batch)
            B, T, C, H, W = radar_recon.shape

            # decode prediction back to 128x128 pixel space
            radar_recon_flat = radar_recon.reshape(B * T, C, H, W)
            radar_recon_dec  = self.decode_stage(self.ae, radar_recon_flat, 1.0)
            radar_recon      = radar_recon_dec.view(B, T, 1, 128, 128)

            radar_ori   = radar_os_gt.cpu().numpy()
            radar_recon = radar_recon.cpu().numpy()

            # evaluate result and save
            if self.is_main:
                eval.evaluate(radar_ori, radar_recon)

            # ===================== PLOTTING =====================                # <<< PLOT
            if do_plot and self.is_main and (batch_idx % plot_stride == 0):       # <<< PLOT
                radar_os_input_np = radar_os_input.cpu().numpy()                  # <<< PLOT
                                                                                # <<< PLOT
                for b in range(B):                                                # <<< PLOT
                    sid = sample_counter + b                                       # <<< PLOT
                                                                                # <<< PLOT
                    # --- Input: (T_in, C, H, W) -> (T_in, H, W) ---             # <<< PLOT
                    inp = radar_os_input_np[b].squeeze(1)  # (5, 128, 128)       # <<< PLOT
                    inp_colored = denorm_and_colorize(                            # <<< PLOT
                        inp, self.scale_value, self.gray2color_fn                 # <<< PLOT
                    )                                                              # <<< PLOT
                    plot_image_sequence_colored(                                   # <<< PLOT
                        inp_colored,                                               # <<< PLOT
                        osp.join(input_dir, f"Sample_{sid}.png"),                  # <<< PLOT
                    )                                                              # <<< PLOT
                                                                                # <<< PLOT
                    # --- Ground Truth: subsample 10 from 20 ---                  # <<< PLOT
                    gt = radar_ori[b].squeeze(1)                                  # <<< PLOT
                    gt_sub = subsample_frames(gt, target_count=10)                # <<< PLOT
                    gt_colored = denorm_and_colorize(                             # <<< PLOT
                        gt_sub, self.scale_value, self.gray2color_fn              # <<< PLOT
                    )                                                              # <<< PLOT
                    plot_image_sequence_colored(                                   # <<< PLOT
                        gt_colored,                                                # <<< PLOT
                        osp.join(gt_dir, f"Sample_{sid}.png"),                     # <<< PLOT
                    )                                                              # <<< PLOT
                                                                                # <<< PLOT
                    # --- Predicted: subsample 10 from 20 ---                     # <<< PLOT
                    pred = radar_recon[b].squeeze(1)                              # <<< PLOT
                    pred_sub = subsample_frames(pred, target_count=10)            # <<< PLOT
                    pred_colored = denorm_and_colorize(                           # <<< PLOT
                        pred_sub, self.scale_value, self.gray2color_fn            # <<< PLOT
                    )                                                              # <<< PLOT
                    plot_image_sequence_colored(                                   # <<< PLOT
                        pred_colored,                                              # <<< PLOT
                        osp.join(pred_dir, f"Sample_{sid}.png"),                   # <<< PLOT
                    )                                                              # <<< PLOT
                                                                                # <<< PLOT
                print(f"[Plot] Saved batch {batch_idx} "                          # <<< PLOT
                    f"(samples {sample_counter}–{sample_counter+B-1})")         # <<< PLOT
            # ===================== END PLOTTING =================                # <<< PLOT

            sample_counter += B                                                   # <<< PLOT

            self.accelerator.wait_for_everyone()
            valid_nums += 1
            if not do_test and self.args.valid_limit and valid_nums >= self.args.vlnum:
                break

        # test done
        if self.is_main:
            res = eval.done()
            if self.is_main and self.args.eval:
                from utils.results_logger_csv import ResultsLogger
                logger = ResultsLogger(csv_path="/home/vatsal/Dataserver2/ECCV26/eval_results.csv")
                logger.log_results(
                    res_dict=res,
                    backbone=self.args.backbone,
                    exp_note=self.args.exp_note,
                    dataset=self.args.dataset,
                )

            prefix = "test" if do_test else "val"
            log_data = {f"{prefix}/{k}": v for k, v in res.items()}
            log_data[f"{prefix}/epoch"] = epoch

            if do_test:
                print_log(f"Test Results: {res}")
            else:
                print_log(f"Valid Results: {res}")
            print_log("=" * 30)

            self.accelerator.log(log_data, step=self.cur_step)

            if self.args.valid:
                return res['csi']
        else:
            return None


        
    def check_milestones(self, target_ckpt=None):
        
        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples_with_plotting(saved_dir_name, do_test=True)
            print("Testing done")
            return
        
        # In case of multiple milestones.
        mils_paths = os.listdir(self.ckpt_path)
        milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)

        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples_with_plotting(milestones[m], do_test=True)
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