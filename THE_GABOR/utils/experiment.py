"""
Shared training harness for Experiment 1.

Scope of this experiment: measure whether models trained on different SEVIR
precipitation regimes end up with different learned Gabor parameter
distributions and functional responses.  No parameter transfer, no freezing,
no cross-regime initialisation, no climate-variable analysis.

Training objective is FACL and nothing else (see models/gabor_mlp_model.py).
"""

import argparse
import json
import os
import os.path as osp
import shutil
import time

import numpy as np
import torch
from tqdm import tqdm

from THE_GABOR.models.gabor_mlp_model import get_model
from THE_GABOR.utils import gabor_logging as glog
from THE_GABOR.utils import gabor_visualization as gviz
from THE_GABOR.utils.gabor_probe import (PROBE_NUM_POINTS, PROBE_SPAN,
                                         build_probe)
from THE_GABOR.utils.init_checkpoint import (architecture_signature,
                                             create_initial_checkpoint,
                                             initial_checkpoint_path,
                                             load_initial_checkpoint,
                                             seed_everything, sha256_file)

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
THE_GABOR_ROOT = osp.join(REPO_ROOT, 'THE_GABOR')

# Placeholder-style configurable paths, resolved from the existing repository.
DEFAULT_OUTPUT_ROOT = osp.join(THE_GABOR_ROOT, 'checkpoints')
DEFAULT_LOG_ROOT = osp.join(THE_GABOR_ROOT, 'logs')
DEFAULT_INIT_ROOT = osp.join(THE_GABOR_ROOT, 'checkpoints', '_initial')
DEFAULT_AE_CKPT = osp.join(REPO_ROOT, 'Pretrained_ae_checkpoints',
                           'autoencoder_checkpoint_32_SEVIR.pth')


# ============================================================
# Arguments
# ============================================================

def base_parser(space):
    p = argparse.ArgumentParser()

    # ---- experiment identity ----
    p.add_argument('--regime', type=str, default='random',
                   choices=['random', 'storm', 'all'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--run_name', type=str, default=None)
    p.add_argument('--wandb_project', type=str, default='THE_GABOR')
    p.add_argument('--wandb_state', type=str, default='online',
                   choices=['online', 'offline', 'disabled'])
    p.add_argument('--output_root', type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--log_root', type=str, default=DEFAULT_LOG_ROOT)
    p.add_argument('--init_root', type=str, default=DEFAULT_INIT_ROOT)
    p.add_argument('--data_root', type=str, default=None,
                   help='override the SEVIR root from datasets/get_datasets.py')

    # ---- data ----
    p.add_argument('--dataset', type=str,
                   default='sevir' if space == 'pixel' else 'sevir_lr_latent_32')
    p.add_argument('--img_size', type=int, default=128 if space == 'pixel' else 32)
    p.add_argument('--img_channel', type=int, default=1 if space == 'pixel' else 4)
    p.add_argument('--frames_in', type=int, default=5)
    p.add_argument('--frames_out', type=int, default=20)
    p.add_argument('--seq_len', type=int, default=25)
    p.add_argument('--stride', type=int, default=13)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=8)

    # ---- controlled architecture ----
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--wave', type=str, default='db6')
    p.add_argument('--wavelet_level', type=int, default=2, choices=[1, 2, 3, 4])
    p.add_argument('--hf_mode', type=str, default='separate',
                   choices=['shared', 'separate'])
    p.add_argument('--size_factor', type=float, default=1.0)

    # ---- Gabor initialisation (identical for every subband, no regime prior) ----
    p.add_argument('--freq_multiplier', type=float, default=1.0)
    p.add_argument('--weight_scale', type=float, default=0.1)
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--beta', type=float, default=1.0)

    # ---- optimisation ----
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_beta1', type=float, default=0.90)
    p.add_argument('--lr_beta2', type=float, default=0.95)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--facl_const_ratio', type=float, default=0.1)

    # ---- Gabor logging cadence ----
    p.add_argument('--gabor_scalar_every_steps', type=int, default=100,
                   help='steps between Gabor scalar-parameter logs')
    p.add_argument('--gabor_probe_every_epochs', type=int, default=1,
                   help='epochs between deterministic probe curve logs')
    p.add_argument('--gabor_hist_every_epochs', type=int, default=5,
                   help='epochs between freq/gamma histograms')
    p.add_argument('--probe_neurons', type=int, default=4)
    p.add_argument('--probe_points', type=int, default=PROBE_NUM_POINTS)
    p.add_argument('--probe_span', type=float, default=PROBE_SPAN)

    # ---- validation / debugging ----
    p.add_argument('--val_every_epochs', type=int, default=5)
    p.add_argument('--limit_train_batches', type=int, default=0)
    p.add_argument('--limit_val_batches', type=int, default=0)
    p.add_argument('--multi_gpu', action='store_true',
                   help='split each batch across all visible GPUs (DataParallel). '
                        'batch_size is the TOTAL batch; per-GPU is batch_size/n_gpu.')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--dry_run', action='store_true',
                   help='build everything, run a couple of steps, then stop')
    return p


# ============================================================
# Experiment
# ============================================================

class GaborExperiment:
    space = None            # 'pixel' | 'latent'

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if args.run_name is None:
            args.run_name = (f'Gabor_{self.space}_SEVIR_{args.regime}'
                             f'_seed{args.seed}')
        self.run_dir = osp.join(args.output_root, args.run_name)
        self.ckpt_dir = osp.join(self.run_dir, 'checkpoints')
        self.plot_dir = osp.join(args.log_root, args.run_name, 'gabor_plots')
        self.probe_dir = osp.join(args.log_root, args.run_name, 'gabor_probe')
        for d in (self.run_dir, self.ckpt_dir, self.plot_dir, self.probe_dir):
            os.makedirs(d, exist_ok=True)

        seed_everything(args.seed)

        # ---- data ----
        self.build_data()

        # ---- model with the shared initial checkpoint ----
        # `limit_train_batches` caps the optimiser steps per epoch.  It is what
        # makes the RANDOM and STORM runs comparable step-for-step: the two
        # regimes have very different numbers of events, so equal epochs would
        # otherwise mean very different numbers of gradient updates.
        if args.limit_train_batches:
            self.steps_per_epoch = min(self.steps_per_epoch, args.limit_train_batches)
        self.total_steps = max(1, args.epochs * self.steps_per_epoch)
        self.data_stats['data/steps_per_epoch'] = int(self.steps_per_epoch)
        self.data_stats['data/total_optimizer_steps'] = int(self.total_steps)
        self.cfg = self.wandb_config()
        self.signature = architecture_signature(self.cfg)
        self.init_path = initial_checkpoint_path(
            args.init_root, self.space, args.seed, self.signature)
        if not osp.exists(self.init_path):
            print(f'[init] {self.init_path} missing -- creating it now. '
                  f'The other regime will load this same file.')
            create_initial_checkpoint(self.build_model, self.init_path,
                                      args.seed, self.cfg)

        self.model = self.build_model().to(self.device)
        load_initial_checkpoint(self.model, self.init_path, self.signature)
        self.init_sha = sha256_file(self.init_path)
        shutil.copyfile(self.init_path, osp.join(self.ckpt_dir, 'initial_model.pt'))
        with open(osp.join(self.run_dir, 'initial_checkpoint.json'), 'w') as f:
            json.dump({'path': self.init_path, 'sha256': self.init_sha,
                       'signature': self.signature}, f, indent=2)
        # Hook for donor-parameter transfer / freezing.  Runs AFTER the shared
        # initial checkpoint is loaded and BEFORE the optimizer is built, so
        # frozen tensors are excluded from the optimizer by construction.
        self.after_init_load()

        self.num_params = sum(p.numel() for p in self.model.parameters()
                              if p.requires_grad)
        print(f'[init] shared initial checkpoint: {self.init_path}')
        print(f'[init] sha256={self.init_sha}  trainable params={self.num_params}')

        # ---- optional multi-GPU ----
        # Only the inner network is wrapped; `self.model` stays unwrapped so that
        # gabor_layers(), checkpointing and Gabor logging keep working on plain
        # module names (no 'module.' prefix ever enters a checkpoint).
        self.dp = None
        inner = getattr(self.model, 'net', None) or getattr(self.model, 'dawncast', None)
        if args.multi_gpu and torch.cuda.device_count() > 1 and inner is not None:
            self.dp = torch.nn.DataParallel(inner)
            print(f'[multi-gpu] DataParallel over {torch.cuda.device_count()} GPUs; '
                  f'total batch {args.batch_size} '
                  f'({args.batch_size // torch.cuda.device_count()} per GPU)')
        elif args.multi_gpu:
            print(f'[multi-gpu] requested but only {torch.cuda.device_count()} GPU visible '
                  f'-- running single-GPU')

        # ---- optimiser ----
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.lr, betas=(args.lr_beta1, args.lr_beta2),
            weight_decay=args.weight_decay)
        from diffusers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.2 * self.total_steps),
            num_training_steps=self.total_steps)

        # ---- fixed deterministic probe ----
        self.probe_s, self.probe_x = build_probe(
            args.frames_in, num_points=args.probe_points, span=args.probe_span)

        # ---- W&B ----
        self.wandb = None
        if not args.no_wandb:
            import wandb
            self.wandb = wandb
            wandb.init(project=args.wandb_project, name=args.run_name,
                       mode=args.wandb_state, config=self.cfg,
                       dir=osp.join(args.log_root, args.run_name))

        self.cur_step = 0
        self.best_metric = -float('inf')

    # ---------------- to be provided by the space-specific runner ----------
    def after_init_load(self):
        """Optional hook: transfer / freeze parameters. Default: no-op."""
        return

    def build_data(self):
        raise NotImplementedError

    def get_seq(self, batch):
        """batch -> (frames_in, frames_gt) on device."""
        raise NotImplementedError

    def validate(self):
        """Returns (metric_for_model_selection, dict_of_metrics) or (None, {})."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    def build_model(self):
        a = self.args
        return get_model(
            T_in=a.frames_in, T_out=a.frames_out, img_channels=a.img_channel,
            dim=a.hidden_dim, weight_scale=a.weight_scale, alpha=a.alpha,
            beta=a.beta, freq_multiplier=a.freq_multiplier,
            size_factor=a.size_factor, wave=a.wave,
            wavelet_level=a.wavelet_level, hf_mode=a.hf_mode,
            total_steps=self.total_steps, const_ratio=a.facl_const_ratio)

    def wandb_config(self):
        a = self.args
        cfg = {
            'model': getattr(self, 'model_name', 'GaborMLPControlled'),
            'dataset': a.dataset,
            'space': self.space,
            'regime': a.regime,
            'seed': a.seed,
            'wavelet': a.wave,
            'wave': a.wave,
            'wavelet_level': a.wavelet_level,
            'hf_mode': a.hf_mode,
            'hidden_dim': a.hidden_dim,
            'frames_in': a.frames_in,
            'frames_out': a.frames_out,
            'seq_len': a.seq_len,
            'stride': a.stride,
            'img_size': a.img_size,
            'img_channel': a.img_channel,
            'batch_size': a.batch_size,
            'epochs': a.epochs,
            'lr': a.lr,
            'freq_multiplier': a.freq_multiplier,
            'weight_scale': a.weight_scale,
            'alpha': a.alpha,
            'beta': a.beta,
            'size_factor': a.size_factor,
            'facl_const_ratio': a.facl_const_ratio,
            'FACL_only': True,
        }
        cfg.update(getattr(self, 'data_stats', {}))
        return cfg

    # ----------------------------------------------------------------------
    def log(self, payload, step=None):
        if self.wandb is not None:
            self.wandb.log(payload, step=self.cur_step if step is None else step)

    def log_gabor_scalars(self):
        self.log(glog.gabor_parameter_scalars(self.model))

    def log_gabor_histograms(self):
        if self.wandb is not None:
            self.log(glog.gabor_histograms(self.model, self.wandb))

    def log_gabor_probe(self, tag):
        """Log the three distinct Gabor quantities on the fixed probe."""
        curves = glog.gabor_probe_curves(self.model, self.probe_s, self.probe_x,
                                         self.args.probe_neurons)
        glog.save_probe_npz(curves, self.probe_dir, tag)
        figures = gviz.make_gabor_figures(curves, tag, save_dir=self.plot_dir)
        # Neuron-averaged panels + scalar summaries.  The scalars are the ones
        # W&B can overlay across runs, so they are what makes RANDOM vs STORM
        # directly comparable on a single chart.
        figures.update(gviz.make_mean_figures(curves, tag, save_dir=self.plot_dir))
        if self.wandb is not None:
            payload = {k: self.wandb.Image(v) for k, v in figures.items()}
            payload.update(glog.mean_curve_summaries(curves))
            self.log(payload)
        gviz.close_figures(figures)

    def save_checkpoint(self, name):
        torch.save({'model': self.model.state_dict(),
                    'step': self.cur_step,
                    'signature': self.signature,
                    'config': self.cfg},
                   osp.join(self.ckpt_dir, name))

    def save_gabor_state(self, name='gabor_state.pt'):
        torch.save({'gabor': glog.gabor_state_dict(self.model),
                    'space': self.space,
                    'regime': self.args.regime,
                    'seed': self.args.seed,
                    'step': self.cur_step,
                    'signature': self.signature},
                   osp.join(self.ckpt_dir, name))

    # ----------------------------------------------------------------------
    def _facl(self):
        return getattr(self.model, 'facl', None) or getattr(self.model, 'falfcl')

    def train_step(self, batch):
        frames_in, frames_gt = self.get_seq(batch)
        if self.dp is not None:
            # forward split across GPUs; FACL computed once on the gathered output
            pred = self.dp(frames_in)
            facl_t = self._facl()(pred, frames_gt)
            loss = {'facl_loss': facl_t, 'total_loss': facl_t}
        else:
            _, loss = self.model.predict(frames_in=frames_in, frames_gt=frames_gt,
                                         compute_loss=True)
        facl = loss['facl_loss']
        total = loss['total_loss']
        # FACL-only guarantee: identical object and numerically equal.
        assert total is facl, 'total_loss must be exactly the FACL loss'
        assert torch.equal(total.detach(), facl.detach())

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        if self.args.grad_clip and self.args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.args.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        self.optimizer.step()
        self.scheduler.step()
        return {'loss/facl': facl.item(),
                'loss/total': total.item(),
                'loss/facl_minus_total': abs(facl.item() - total.item()),
                'grad_norm': float(grad_norm),
                'lr': self.optimizer.param_groups[0]['lr']}

    def train(self):
        a = self.args
        # checkpoint label 'init' -- state before any optimisation step
        self.log_gabor_scalars()
        self.log_gabor_probe('init')
        self.save_gabor_state('gabor_state_init.pt')

        for epoch in range(a.epochs):
            self.model.train()
            t0 = time.time()
            running, nsteps = 0.0, 0
            pbar = tqdm(self.train_loader, total=self.steps_per_epoch,
                        desc=f'{a.run_name} epoch {epoch + 1}/{a.epochs}')
            for i, batch in enumerate(pbar):
                if a.limit_train_batches and i >= a.limit_train_batches:
                    break
                logs = self.train_step(batch)
                running += logs['loss/total']
                nsteps += 1
                self.cur_step += 1
                self.log(logs)
                if a.gabor_scalar_every_steps and \
                        self.cur_step % a.gabor_scalar_every_steps == 0:
                    self.log_gabor_scalars()
                if a.dry_run and self.cur_step >= 3:
                    break

            self.log({'epoch/total_loss': running / max(nsteps, 1),
                      'epoch/facl_loss': running / max(nsteps, 1),
                      'epoch/index': epoch + 1,
                      'epoch/seconds': time.time() - t0})
            self.log_gabor_scalars()
            if a.gabor_probe_every_epochs and \
                    (epoch + 1) % a.gabor_probe_every_epochs == 0:
                self.log_gabor_probe(f'epoch_{epoch + 1:03d}')
            if a.gabor_hist_every_epochs and \
                    (epoch + 1) % a.gabor_hist_every_epochs == 0:
                self.log_gabor_histograms()

            if a.dry_run:
                break

            if a.val_every_epochs and (epoch + 1) % a.val_every_epochs == 0:
                metric, metrics = self.validate()
                self.log({f'val/{k}': v for k, v in metrics.items()})
                self.log({'val/epoch': epoch + 1})
                if metric is not None and metric > self.best_metric:
                    self.best_metric = metric
                    self.save_checkpoint('best_model.pt')
                    self.save_gabor_state('gabor_state_best.pt')
                    print(f'[val] new best metric {metric:.5f} -- best_model.pt saved')
            self.save_checkpoint('last_model.pt')

        self.log_gabor_probe('final')
        self.save_checkpoint('final_model.pt')
        self.save_gabor_state('gabor_state.pt')
        if not osp.exists(osp.join(self.ckpt_dir, 'best_model.pt')):
            # no validation ran (e.g. dry run) -- keep the required filename
            self.save_checkpoint('best_model.pt')
        print(f'[done] checkpoints in {self.ckpt_dir}')
        if self.wandb is not None:
            self.wandb.finish()
