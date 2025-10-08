import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os.path as osp
import math
import time
import argparse
import logging 
import yaml
import cProfile
from tqdm import tqdm
from datetime import timedelta
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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
from datasets.dataset_mosdac import *
from datasets.get_datasets import get_dataset
from utils.metrics import Evaluator
from utils.tools import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm, BoundaryNorm
import matplotlib.colors as mcolors
import numpy as np
from copy import deepcopy


# Apply your own wandb api key to log online
# os.environ["WANDB_API_KEY"] = "ebdff79c224117070aea32ad36c6031428ab5f04"
# os.environ["WANDB_SILENT"] = "true"
os.environ["ACCELERATE_DEBUG_MODE"] = "1"


def create_parser():
    # --------------- Basic ---------------
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--backbone',        type=str,            default='phydnet',              help='backbone model for deterministic prediction (earthformer, simvp, phydnet)')
    parser.add_argument('--use_diff',        action="store_true", default=False,                  help='Weather use diff framework, as for ablation study')
    parser.add_argument("--seed",           type=int,             default=0,                      help='Experiment seed')
    parser.add_argument("--exp_dir",        type=str,             default='sevir',           help="experiment directory")
    parser.add_argument("--exp_note",       type=str,             default=None,                   help="additional note for experiment")


    # --------------- Dataset ---------------
   
    parser.add_argument("--file_rain_seq_add",  type=str,   default=0,                  help="Rainy days file")
    parser.add_argument("--method",             type= int,  default= None,              help = "Method to select the dataset as per the need. (Look at the function for more details)")
    parser.add_argument("--dataset",            type=str,   default='sevir',            help="dataset name, use 'vil_mosdac' for vil_scaled dataset and 'mosdac' for reflectivity dataset.")
    parser.add_argument("--img_size",           type=int,   default=128,                help="image size")
    parser.add_argument("--img_channel",        type=int,   default=1,                  help="channel of image")
    parser.add_argument("--stride",             type=int,   default=13,                 help="dataset stride")
    parser.add_argument("--seq_len",            type=int,   default=25,                 help="sequence length sampled from dataset")
    parser.add_argument("--frames_in",          type=int,   default=5,                  help="number of frames to input")
    parser.add_argument("--frames_out",         type=int,   default=20,                 help="number of frames to output")    
    parser.add_argument("--num_workers",        type=int,   default=4,                  help="number of workers for data loader")
    parser.add_argument("--preprocessing",      type=int,   default=None,               help="Type to preprocess the data")
    
    # --------------- Optimizer ---------------
    parser.add_argument("--lr",             type=float, default=1e-5,            help="learning rate")
    parser.add_argument("--lr-beta1",       type=float, default=0.90,            help="learning rate beta 1")
    parser.add_argument("--lr-beta2",       type=float, default=0.95,            help="learning rate beta 2")
    parser.add_argument("--l2-norm",        type=float, default=0.0,             help="l2 norm weight decay")
    parser.add_argument("--ema_rate",       type=float, default=0.95,            help="exponential moving average rate")
    parser.add_argument("--scheduler",      type=str,   default='cosine',        help="learning rate scheduler", choices=['constant', 'linear', 'cosine'])
    parser.add_argument("--warmup_steps",   type=int,   default=1000,            help="warmup steps")
    parser.add_argument("--mixed_precision",type=str,   default='no',            help="mixed precision training")
    parser.add_argument("--grad_acc_step",  type=int,   default=1,               help="gradient accumulation step")
    
    # --------------- Training ---------------
    parser.add_argument("--batch_size",     type=int,   default=4,               help="batch size")
    parser.add_argument("--epochs",         type=int,   default=20,              help="number of epochs")
    parser.add_argument("--early_stop",     type=int,   default=10,              help="early stopping steps")
    parser.add_argument("--ckpt_milestone", type=str,   default=None,            help="resumed checkpoint milestone")
    parser.add_argument("--datatype",       type=str, default=None,                   help="Indicates the datatype available (reflectivity, vil, vil_vip)")
    # --------------- Additional Ablation Configs ---------------
    parser.add_argument("--eval",           action="store_true",                 help="evaluation mode")
    
    # --------------- Wandb ---------------
    parser.add_argument("--wandb_state",        type=str,       default="offline",          help="wandb state config")
    parser.add_argument("--wandb_project_name", type=str,       default="Wandb Project",    help="wandb project name")
    parser.add_argument("--run_name",           type=str,       default='run_1',            help="wandb run name")
    
    #------------------------- Plots -----------------------------
    parser.add_argument("--generate_outputs",      action="store_true",          help="Generate visualizations from checkpoint")
    parser.add_argument("--plot_saving_directory", type=str,  default=None,      help="Enter saving directory for plots")

    args = parser.parse_args()
    return args



class Runner(object):
    
    def __init__(self, args):
        
        self.args = args
        self._preparation()
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
            # project_name=self.exp_name,
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
        
        print_log('============================================================', self.is_main)
        print_log("                 Experiment Start                           ", self.is_main)
        print_log('============================================================', self.is_main)
    
        print_log(self.accelerator.state, self.is_main)
        print("Loading data")
        self._load_data()
        print("Building model")
        self._build_model()
        self._build_optimizer()
        
        # distributed ema for parallel sampling

        self.model, self.optimizer,  self.scheduler, self.train_loader, self.valid_loader, self.test_loader = self.accelerator.prepare(
            self.model, 
            self.optimizer, self.scheduler,
            self.train_loader, self.valid_loader, self.test_loader
        )
        
        self.train_dl_cycle = cycle(self.train_loader)
        if self.is_main:
            start = time.time()
            next(self.train_dl_cycle)
            print_log(f"Data Loading Time: {time.time() - start}", self.is_main)
            # print_log(show_img_info(sample), self.is_main)
        print(torch.cuda.is_available())
        print_log(f"gpu_nums: {torch.cuda.device_count()}, gpu_id: {torch.cuda.current_device()}")
        
        if self.args.ckpt_milestone is not None:
            self.load(self.args.ckpt_milestone)

    @property
    def is_main(self):
        return self.accelerator.is_main_process
    
    @property
    def device(self):
        return "cuda:0"
    
    def _preparation(self):
        # =================================
        # Build Exp dirs and logging file
        # =================================

        set_seed(self.args.seed)
        self.model_name = self.model_name = ('Diff' if self.args.use_diff else 'Single') + self.args.backbone
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
        
    def _load_data(self):
        # =================================
        # Get Train/Valid/Test dataloader among datasets 
        # =================================
        # if self.args.dataset == 'custom':
        #     self.args.num_workers = 1
            
        train_data, valid_data, test_data, color_save_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
            data_name=self.args.dataset,
            # data_path=self.args.data_path,
            img_size=self.args.img_size,
            seq_len=self.args.seq_len,
            file_rain_seq_add=self.args.file_rain_seq_add,
            method = self.args.method,
            in_channels = self.args.frames_in,
            out_channels = self.args.frames_out,
            batch_size=self.args.batch_size, 
            stride = self.args.stride,
            preprocess_type = self.args.preprocessing
        )
        
        self.visiual_save_fn = color_save_fn
        self.thresholds      = THRESHOLDS
        self.scale_value     = PIXEL_SCALE
        
        # if self.args.dataset != 'sevir':
        #     # preload big batch data for gradient accumulation
        #     self.train_loader = torch.utils.data.DataLoader(
        #         train_data, batch_size=self.args.batch_size*self.args.grad_acc_step, shuffle=True, num_workers=self.args.num_workers, drop_last=True
        #     )
        #     self.valid_loader = torch.utils.data.DataLoader(
        #         valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers, drop_last=True
        #     )
        #     self.test_loader = torch.utils.data.DataLoader(
        #         test_data, batch_size=self.args.batch_size , shuffle=False, num_workers=self.args.num_workers
        #     )


        if self.args.dataset == 'vil_mosdac' or self.args.dataset == 'vil' or self.args.dataset == 'mosdac':
        
            self.train_loader = create_loader(train_data, batch_size= self.args.batch_size, shuffle=True)
            self.valid_loader = create_loader(valid_data, batch_size= self.args.batch_size)
            self.test_loader = create_loader(test_data, batch_size= self.args.batch_size)

        else:
            self.train_loader = train_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.valid_loader = valid_data.get_torch_dataloader(num_workers=self.args.num_workers)
            self.test_loader = test_data.get_torch_dataloader(num_workers=self.args.num_workers)
            
            
        print_log(f"train data: {len(self.train_loader)}, valid data: {len(self.valid_loader)}, test_data: {len(self.test_loader)}",
                  self.is_main)
    
        print_log(f"Pixel Scale: {PIXEL_SCALE}, Threshold: {str(THRESHOLDS)}", self.is_main)
        for loader in [self.train_loader, self.valid_loader, self.test_loader]:
            for batch in loader:
                from termcolor import colored
                print_log(colored(f"Batch Shape: {batch.shape}, Type: {batch.dtype}", 'green'), self.is_main)
                break
    
    def _build_model(self):
        # =================================
        # import and create different models given model config
        # =================================

        if self.args.backbone == 'simvp':
            from models.simvp import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
            }
            model = get_model(**kwargs)
        
        elif self.args.backbone == 'phydnet':
            from models.phydnet import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
                "device": self.device
            }
            model = get_model(**kwargs)

        elif self.args.backbone == 'earthformer':
            from models.earth_former import get_model
            kwargs = {
                "in_shape": (self.args.img_channel, self.args.img_size, self.args.img_size),
                "T_in": self.args.frames_in,
                "T_out": self.args.frames_out,
            }
            model = get_model(**kwargs)
        
        elif self.args.backbone == 'fno':
            from models.fno import FNOModel
            model = FNOModel(in_channels=self.args.frames_in, out_channels=self.args.frames_out)
            
        else:
            raise NotImplementedError
        
        if self.args.use_diff:
            from models.diffcast import get_model
            kwargs = {
                'img_channels' : self.args.img_channel,
                'dim' : 64,
                'dim_mults' : (1,2,4,8),
                'T_in': self.args.frames_in,
                'T_out': self.args.frames_out,
                'sampling_timesteps' : 250,
            }
            diff_model = get_model(self.args.img_channel, 64, (1,2,4,8), self.args.frames_in, self.args.frames_out, 1000, sampling_timesteps=250)
            diff_model.load_backbone(model)
            model = diff_model

            
        self.model = model
        print_log("begin ema", self.is_main)
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)        
        print_log("end device", self.is_main)
        if self.is_main:
            total = sum([param.nelement() for param in self.model.parameters()])
            print_log("Main Model Parameters: %.2fM" % (total/1e6), self.is_main)


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
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    self.args.scheduler
            )
        )
            
        if self.is_main:
            print_log("============ Running training ============")
            print_log(f"    Num examples = {len(self.train_loader)}")
            print_log(f"    Num Epochs = {self.global_epochs}")
            print_log(f"    Instantaneous batch size per GPU = {self.args.batch_size}")
            print_log(f"    Total train batch size (w. parallel, distributed & accumulation) = {self.args.batch_size * self.accelerator.num_processes}")
            print_log(f"    Total optimization steps = {self.global_steps}")
            print_log(f"    Optimizer: {self.optimizer} with init lr: {self.args.lr}")
        
    
    def save(self):
        # =================================
        # Save checkpoint stsate for model and ema
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
        
        torch.save(data, osp.join(self.ckpt_path, f"ckpt-{self.cur_step}.pt"))
        print("Checkpoint saved")
        print_log(f"Save checkpoint {self.cur_step} to {self.ckpt_path}", self.is_main)
        
        
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
        
        self.optimizer.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])
        
        if self.is_main:
            self.ema.load_state_dict(data['ema'])

        # self.cur_epoch = data['epoch']
        # self.cur_step = data['step']
        print_log(f"Load checkpoint {milestone} from {self.ckpt_path}", self.is_main)
        
    
    def train(self):
        print("Training Now")
        # set global step as traing process
        pbar = tqdm(
            initial=self.cur_step,
            total=self.global_steps,
            # disable=not self.is_main,
            disable = True
        )
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):

            epoch_start_time = time.time()
            self.cur_epoch = epoch
            print("Training model")
            self.model.train()
            
            
            for i, batch in enumerate(self.train_loader):
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
                    log_dict[k] = v.item()
                self.accelerator.log(log_dict, step=self.cur_step)
                # pbar.set_postfix(**log_dict)   
                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"
                # pbar.set_description(state_str)
                
                # update ema param and log file every 20 steps
                if i % 50 == 0:
                    print(state_str+'::'+str(log_dict))
                self.ema.update()

                self.cur_step += 1
                pbar.update(1)
                
                # test and log metrics every 10000 steps
                # if self.cur_step % 10000 == 0:
                #     print_log(f" ========= Running Test at Step {self.cur_step} ==========", self.is_main)
                #     self.test_samples(self.cur_step, do_test=False)  # Run validation test
                #     self.model.train()  # Set back to training mode
                
                # do santy check at begining
                if self.cur_step == 1:
                    """ santy check """
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon= self._sample_batch(batch)
                            from termcolor import colored
                            print_log(colored(f"Sanity Check: {radar_ori.shape}, {radar_recon.shape}", 'blue'), self.is_main)
                            os.makedirs(self.sanity_path)
                            if self.is_main:
                                for i in range(radar_ori.shape[0]):
                                    self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(self.sanity_path, f"{i}/reflectivity"),data_type='vil')
                            print_log(f" ========= Sanity Check Completed==========", self.is_main)

                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)

            # save checkpoint and do test every epoch
            if epoch==0 or (epoch+1)%10==0:
                self.save()
                self.test_samples(epoch, self.cur_step, do_test=False)  # Run validation test
                self.model.train()  # Set back to training mode
            
            epoch_time = time.time() - epoch_start_time
            print_log(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds.")
            print_log(f" ========= Finished one Epoch ==========Epoch number:{epoch+1}==========", self.is_main)

        self.accelerator.end_training()
        
    def _get_seq_data(self, batch):
        # frame_seq = batch['vil'].unsqueeze(2).to(self.device)
        return batch      # [B, T, C, H, W]
    
    def _train_batch(self, batch):
        radar_batch = self._get_seq_data(batch)
        frames_in, frames_out = radar_batch[:,:self.args.frames_in], radar_batch[:,self.args.frames_in:]

        assert radar_batch.shape[1] == self.args.frames_out + self.args.frames_in, "radar sequence length error"
        _, loss = self.model.predict(frames_in=frames_in, frames_gt=frames_out, compute_loss=True)
        if loss is None:
            raise ValueError("Loss is None, please check the model predict function")
        return {'total_loss': loss}
        
    
    @torch.no_grad()
    def _sample_batch(self, batch, use_ema=False):
        sample_fn = self.ema.ema_model.predict if use_ema else self.model.predict
        frame_in = self.args.frames_in
        radar_batch = self._get_seq_data(batch)
        radar_input, radar_gt = radar_batch[:,:frame_in], radar_batch[:,frame_in:]
        radar_pred, _ = sample_fn(radar_input,compute_loss=False)
        
        
        radar_gt = self.accelerator.gather(radar_gt).detach().cpu().numpy()

        radar_pred = self.accelerator.gather(radar_pred).detach().cpu().numpy()
        return radar_gt, radar_pred
    
    # Doing do_test -> True because I need lesser samples 
    def test_samples(self, milestone,epoch = None, do_test=False):
        if do_test==False:
            print("Validation")
        if do_test==True:
            print("Testing")
        # init test data loader

        data_loader = self.test_loader if do_test else self.valid_loader
        # init sampling method
        self.model.eval()
        # init test dir config
        # cnt = 0 # This will be replaced by batch_idx from enumerate
        save_dir = osp.join(self.test_path, f"sample-{milestone}") if do_test else osp.join(self.valid_path, f"sample-{milestone}")
        os.makedirs(save_dir, exist_ok=True)
        if self.is_main:
            eval = Evaluator(
                seq_len=self.args.frames_out,
                value_scale=self.scale_value,
                thresholds=self.thresholds,
                save_path=save_dir,
            )

        """
        Radar ori , radar recon shape: (3, 20, 1, 240, 240)
        """
        # start test loop
        for batch_idx, batch in enumerate(tqdm(data_loader,desc='Test Samples', disable=not self.is_main)):
            # sample
            radar_ori, radar_recon= self._sample_batch(batch)
            # print(f"Radar ori, shape:{radar_ori.shape}, type: {type(radar_ori)}, Radar recon, shape:{radar_recon.shape}, type: {type(radar_recon)}")
           
            # evaluate result and save
            if self.is_main: # Evaluator and visiual_save_fn should only run on main process
                eval.evaluate(radar_ori, radar_recon)
                # for i in range(radar_ori.shape[0]): # Iterate over samples in the batch
                    # Use batch_idx to ensure unique paths for samples from different batches
                    # self.visiual_save_fn(radar_recon[i], radar_ori[i], osp.join(save_dir, f"{batch_idx}-{i}/vil"),data_type='vil')

            self.accelerator.wait_for_everyone()
            # cnt += 1 # Not needed if using batch_idx
            # if cnt > 10: # If you want to limit the number of batches processed for testing
            #     break
        # test done
        if self.is_main:
            res = eval.done()
            if do_test:
                print_log(f"Test Results: {res}")
            else:
                print_log(f"Valid Results: {res}")
            print_log("="*30)
            # Log to wandb
            self.accelerator.log({**res, "epoch": epoch}, step=self.cur_step)

    # def convert_vip_vil(self, np_array):
    #     if np_array<=5:

    def _extract_date_from_fname(fname):
        # fname can be '01AUG2023_023340.npy' or '01AUG2023_023340'
        base = os.path.basename(fname)
        if base.endswith('.npy'):
            base = base[:-4]
        date_part = base.split('_')[0]
        return date_part       
        


    def check_milestones(self, target_ckpt=None):

        mils_paths = os.listdir(self.ckpt_path)
        milestones = sorted([int(m.split('-')[-1].split('.')[0]) for m in mils_paths], reverse=True)
        print_log(f"milestones: {milestones}", self.accelerator.is_main_process)
        
        if target_ckpt is not None:
            self.load(target_ckpt)
            saved_dir_name = target_ckpt.split('/')[-1].split('.')[0]
            self.test_samples(saved_dir_name, do_test=True)
            print("Testing done")
            return
        
        for m in range(0, len(milestones), 1):
            self.load(milestones[m])
            self.test_samples(milestones[m], do_test=True)
            break
    
    
    
    def generate_outputs_from_checkpoint(self, save_dir=None, data_type=None, target_ckpt=None):
        

        if data_type == None or save_dir == None:
            KeyError("datatype or save_dir not defined.")
        checkpoint_path= target_ckpt

        device = self.device
        self.load(checkpoint_path)

        self.model.eval()

        latitude_range = slice(6.37081, 10.71)
        longitude_range = slice(74.68674, 79.02363)

        # Number of steps
        lat_steps = self.args.img_size
        lon_steps = self.args.img_size

        # Generate coordinate arrays
        lat_range = np.linspace(latitude_range.start, latitude_range.stop, lat_steps)
        lon_range = np.linspace(longitude_range.start, longitude_range.stop, lon_steps)

        

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, "input")
        gt_dir = os.path.join(save_dir, "gt")
        result_dir = os.path.join(save_dir, "result")

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)

        for sample_idx, batch in enumerate(self.test_loader):
            if sample_idx%1 == 0:
                x_test, y_test = batch[:, :self.args.frames_in], batch[:, self.args.frames_in:]
                
                x_test, y_test = x_test.to(device), y_test.to(device)
                
                with torch.no_grad():
                    preds, _ = self.model.predict(x_test, y_test, compute_loss=False)

                # x_test = preprocess_input(x_test)
                # y_test = preprocess_input(y_test)
           
                x_np = x_test[0].squeeze(1).cpu().numpy()
                y_np = y_test[0].squeeze(1).cpu().numpy()
                p_np = preds[0].squeeze(1).cpu().numpy()

###################################################################################################################################################
                                                #  Convert back to normal.
###################################################################################################################################################
                # x_np, y_np, p_np = unnormalize(x_np, min= 0, max = 76377.62), unnormalize(y_np, min= 0, max = 76377.62), unnormalize(p_np, min= 0, max = 76377.62)
                
                if data_type.lower() == "reflectivity":
                    x_np, p_np, y_np = 60*x_np, 60*p_np, 60*y_np

                    vil_colors = [ "#f0f0f0", "#00b0f0", "#00ff80", "#a0f000", "#f0d000", "#f08000", "#f00000", "#c00000", "#2020ff", "#a000a0", "#000000"]
                    vil_cmap = ListedColormap(vil_colors, name='vil_colormap')
                    norm_vil = mcolors.Normalize(vmin=0, vmax=60)

                elif data_type.lower() == "vil_vip":
                    x_np, p_np, y_np = 255*x_np, 255*p_np, 255*y_np
                    VIL_COLORS = [[0, 0, 0],
                    [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
                    [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
                    [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
                    [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
                    [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
                    [0.9607843137254902, 0.9607843137254902, 0.0],
                    [0.9294117647058824, 0.6745098039215687, 0.0],
                    [0.9411764705882353, 0.43137254901960786, 0.0],
                    [0.6274509803921569, 0.0, 0.0],
                    [0.9058823529411765, 0.0, 1.0]]

                    VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
                    VIL_LEVELS = [0.0, 0.15, 0.25, 0.52, 0.77, 1.51, 3.53, 7.07, 12.13, 32.23, 81.32]

                    cols = deepcopy(VIL_COLORS)
                    lev = deepcopy(VIL_LEVELS)

                    nil = cols.pop(0)
                    under = cols[0]
                    # over = cols.pop()
                    over = cols[-1]
                    vil_cmap = ListedColormap(cols)
                    vil_cmap.set_bad(nil)
                    vil_cmap.set_under(under)
                    vil_cmap.set_over(over)
                    norm_vil = BoundaryNorm(lev, vil_cmap.N)

                elif data_type.lower() == "vil":
                    peak_val = 5000
                    x_np, p_np, y_np = convert_vip_vil(x_np), convert_vip_vil(p_np), convert_vip_vil(y_np)
                    x_np, p_np, y_np = 255*x_np, 255*p_np, 255*y_np
                    VIL_COLORS = [[0, 0, 0],
                    [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
                    [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
                    [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
                    [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
                    [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
                    [0.9607843137254902, 0.9607843137254902, 0.0],
                    [0.9294117647058824, 0.6745098039215687, 0.0],
                    [0.9411764705882353, 0.43137254901960786, 0.0],
                    [0.6274509803921569, 0.0, 0.0],
                    [0.9058823529411765, 0.0, 1.0]]

                    VIL_LEVELS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, 255.0]
                    VIL_LEVELS = np.linspace(0,int(peak_val), len(VIL_LEVELS))

                    cols = deepcopy(VIL_COLORS)
                    lev = deepcopy(VIL_LEVELS)

                    nil = cols.pop(0)
                    under = cols[0]
                    # over = cols.pop()
                    over = cols[-1]
                    vil_cmap = ListedColormap(cols)
                    vil_cmap.set_bad(nil)
                    vil_cmap.set_under(under)
                    vil_cmap.set_over(over)
                    norm_vil = BoundaryNorm(lev, vil_cmap.N)

                lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
                

                # Plot inputs
                fig_input, axes_input = plt.subplots(1, self.args.frames_in, figsize=(50, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                for i in range(self.args.frames_in):
                    im = axes_input[i].pcolormesh(lon_grid, lat_grid, x_np[i], cmap=vil_cmap,norm = norm_vil, shading='auto')
                    axes_input[i].set_title(f"Input {i+1}", fontsize = 25)
                    axes_input[i].axis("off")
                    axes_input[i].add_feature(cfeature.BORDERS)
                    axes_input[i].add_feature(cfeature.COASTLINE)
                    axes_input[i].add_feature(cfeature.LAND, facecolor='none')

                fig_input.tight_layout()
                # fig_input.colorbar(im, ax=axes_input.ravel().tolist(), orientation='horizontal')
                fig_input.savefig(os.path.join(input_dir, f"Input_Sample{sample_idx}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_input)

                # Plot GT
                fig_gt, axes_gt = plt.subplots(1, 10, figsize=(50, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                for j in range(10):
                    im = axes_gt[j].pcolormesh(lon_grid, lat_grid, y_np[j], cmap=vil_cmap,norm = norm_vil, shading='auto')
                    axes_gt[j].set_title(f"GT {j+1}", fontsize = 25)
                    axes_gt[j].axis("off")
                    axes_gt[j].add_feature(cfeature.BORDERS)
                    axes_gt[j].add_feature(cfeature.COASTLINE)
                    axes_gt[j].add_feature(cfeature.LAND, facecolor='none')

                fig_gt.tight_layout()
                # fig_gt.colorbar(im, ax=axes_gt.ravel().tolist(), orientation='horizontal')
                fig_gt.savefig(os.path.join(gt_dir, f"GT_Sample{sample_idx}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_gt)

                
                fig_result, axes_result = plt.subplots(1, 10, figsize=(50, 10), subplot_kw={'projection': ccrs.PlateCarree()})
                for k in range(10):
                    im = axes_result[k].pcolormesh(lon_grid, lat_grid, p_np[k], cmap=vil_cmap,norm = norm_vil, shading='auto')
                    axes_result[k].set_title(f"Pred_{k+1}", fontsize = 25)
                    axes_result[k].axis("off")
                    axes_result[k].add_feature(cfeature.BORDERS)
                    axes_result[k].add_feature(cfeature.COASTLINE)
                    axes_result[k].add_feature(cfeature.LAND, facecolor='none')

                fig_result.tight_layout()
                # fig_result.colorbar(im, ax=axes_result.ravel().tolist(), orientation='horizontal')
                fig_result.savefig(os.path.join(result_dir, f"Pred_Sample{sample_idx}.png"), dpi=300, bbox_inches='tight')
                plt.close(fig_result)
                
                print(f"Plotted at directory, {result_dir}, Sample_idx; {sample_idx}")

def main():
    args = create_parser()
    exp = Runner(args)
    

    if args.generate_outputs:
        # When just evaluating and visualizing
        save_dir = "/home/vatsal/NWM/weather/DiffCast_mosdac/Plots/VIL_VIP/ISRO/"
        exp.generate_outputs_from_checkpoint(save_dir, data_type = args.datatype, target_ckpt=args.ckpt_milestone)

    else: 
        if not args.eval:
            exp.train()
            # exp.check_milestones()
        else:
            
            exp.check_milestones(target_ckpt=args.ckpt_milestone)

if __name__ == '__main__':
    # 测试代码各模块执行效率
    # pip install graphviz
    # pip install gprof2dot
    # gprof2dot -f pstats train.profile | dot -Tpng -o result.png
    # cProfile.run('main()', filename='train.profile', sort='cumulative')
    main()
