"""
=============================================================================
SEMI-FOUNDATION-MODEL TRANSFER - PIXEL SPACE
=============================================================================

Pixel-space counterpart of finetune_temporal_path_transfer.py: same freeze /
audit / adapter machinery, but on 128x128 single-channel radar instead of
4x32x32 SD-VAE latents, so there is NO AutoencoderKL anywhere in the loop.

Two differences from the latent script, both forced by the checkpoint:

  1. Base runner is run_alphapre_convlstm.py (pixel space), not
     run_alphapre_convlstm_sevir_lr_latent.py.
  2. The pixel SEVIR checkpoint was written by models/DAWNCast/dawncast.py,
     which uses the RENAMED modules, so every pattern shifts:

        latent (dawncast_old.py)          pixel (dawncast.py)
        ------------------------          -------------------
        lastocast.                        dawncast.
        operator                          wgtm
        stream_ll / hf_streams.i          fat_ll  / fat_hf_streams.i
        conv_spectral                     srst
        block1/block2                     srst_block1/srst_block2
        dw_spatial                        spatial_branch

     The freeze logic itself is naming-agnostic (it walks module types), so
     only the audit patterns and the depthwise-conv attribute name change.

Everything else - the unfreeze groups, the exact-match audit, the frozen-norm
eval lock, the step-N frozen-parameter equality check, the adapter checkpoint
and the CSV/WandB logging - is imported unchanged from the latent script.

USAGE:
    CUDA_VISIBLE_DEVICES=0 python finetune_temporal_path_transfer_pixel.py \
        --pretrained_ckpt /home/vatsal/Dataserver2/Neurips/DAWNCast_pixelspace/\
dawncast_sevir_pixel/checkpoints/ckpt-best.pt \
        --unfreeze temporal lifting projection \
        --backbone DAWNCast --dataset meteo \
        --img_size 128 --img_channel 1 --frames_in 5 --frames_out 20 --seq_len 25 \
        --wave db6 --wavelet_level 2 --hf_mode separate --gpu_use 0 --valid
=============================================================================
"""
import os
import os.path as osp
import re
import sys
import argparse

import torch
from torch import nn
from ema_pytorch import EMA

from utils.tools import print_log

# Shared machinery - imported, not duplicated.
import finetune_temporal_path_transfer as LATENT
from finetune_temporal_path_transfer import (
    NORM_TYPES,
    collect_temporal_path_modules,
    audit_trainable,
    lock_frozen_norms,
    snapshot_frozen,
    patch_results_logger,
)

from run_alphapre_convlstm import Runner, create_parser

# =============================================================================
# Patterns for the dawncast.py naming. Patched into the shared module for THIS
# process only - the latent script keeps its own patterns in its own runs.
# =============================================================================
PIXEL_TRAINABLE_RE = re.compile(
    r"^dawncast\.wgtm\."
    r"(fat_ll|fat_hf|fat_hf_streams\.\d+)\."
    r"(gabor\.(mu|freq|gamma|linear\.(weight|bias))|mlp\.\d+\.(weight|bias)|fusion\.(weight|bias))$"
)
PIXEL_GROUP_NAME_RE = {
    'temporal':    PIXEL_TRAINABLE_RE,
    'lifting':     re.compile(r"^dawncast\.lifting\."),
    'projection':  re.compile(r"^dawncast\.projection\."),
    'norms':       re.compile(r"\.norm\.(weight|bias)$"),
    'biases':      re.compile(r"(\.bias|\.b1|\.b2)$"),
    'norms_stem':  re.compile(r"^(?!.*\.srst\.).*\.norm\.(weight|bias)$"),
    'biases_stem': re.compile(r"^(?!.*\.srst\.).*(\.bias|\.b1|\.b2)$"),
    'dw_spatial':  re.compile(r"\.spatial_branch\.weight$"),
}
LATENT.GROUP_NAME_RE = PIXEL_GROUP_NAME_RE

# dawncast.py renames the depthwise spatial conv; the frozen spectral trunk too.
DW_ATTR = 'spatial_branch'
SPECTRAL_MARKERS = ('.srst.', '.conv_spectral.')

PRETRAINED_PIXEL_CKPT = ("/home/vatsal/Dataserver2/Neurips/DAWNCast_pixelspace/"
                         "dawncast_sevir_pixel/checkpoints/ckpt-best.pt")


# =============================================================================
# Group selection (dawncast.py naming)
# =============================================================================

def collect_unfreeze_groups(model, groups):
    """Same contract as the latent version; only the module names differ."""
    sel = {g: {} for g in groups}

    def in_spectral(name):
        return any(m in name for m in SPECTRAL_MARKERS)

    if 'temporal' in sel:
        for mod_name, module in collect_temporal_path_modules(model).items():
            for pn, p in module.named_parameters(recurse=True):
                sel['temporal'][f"{mod_name}.{pn}"] = p

    for group, attr in (('lifting', 'lifting'), ('projection', 'projection'),
                        ('dw_spatial', DW_ATTR)):
        if group not in sel:
            continue
        for mod_name, module in model.named_modules():
            if mod_name.split('.')[-1] != attr:
                continue
            for pn, p in module.named_parameters(recurse=True):
                sel[group][f"{mod_name}.{pn}"] = p

    for g in ('norms', 'norms_stem'):
        if g not in sel:
            continue
        for mod_name, module in model.named_modules():
            if not isinstance(module, NORM_TYPES):
                continue
            if g == 'norms_stem' and in_spectral(mod_name + '.'):
                continue
            for pn, p in module.named_parameters(recurse=False):
                sel[g][f"{mod_name}.{pn}"] = p

    for g in ('biases', 'biases_stem'):
        if g not in sel:
            continue
        for name, p in model.named_parameters():
            if not (name.endswith('.bias') or name.endswith('.b1') or name.endswith('.b2')):
                continue
            if g == 'biases_stem' and in_spectral(name):
                continue
            sel[g][name] = p

    empty = [g for g, d in sel.items() if not d]
    if empty:
        raise RuntimeError(f"Unfreeze group(s) matched no parameters: {empty}")
    return sel


def apply_unfreeze(model, groups):
    """Freeze everything, then re-open exactly the requested groups."""
    for p in model.parameters():
        p.requires_grad = False

    sel = collect_unfreeze_groups(model, groups)
    union = {}
    for members in sel.values():
        for name, p in members.items():
            p.requires_grad = True
            union[name] = p
    return sel, union


# =============================================================================
# Runner
# =============================================================================

class PixelTransferRunner(Runner):
    """Pixel-space Runner with the pretrained load, freeze, audit and checks."""

    def __init__(self, args, ft_args):
        self.ft_args = ft_args
        self.frozen_snapshot = None
        self.trainable_names = []
        self.freeze_check_done = False
        super().__init__(args)

    # ----------------------------------------------------------------- data --
    def _load_data(self):
        super()._load_data()

        frac = self.ft_args.train_frac
        if frac >= 1.0:
            return

        dataset = self.train_loader.dataset
        n_total = len(dataset)
        n_keep = max(1, int(round(n_total * frac)))
        gen = torch.Generator().manual_seed(self.args.seed)
        keep = torch.randperm(n_total, generator=gen)[:n_keep].tolist()

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, keep),
            batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.num_workers, drop_last=True,
        )
        print_log(f"Train subsample: {frac:.0%} -> {n_keep}/{n_total} sequences, "
                  f"{len(self.train_loader)} batches/epoch (seed {self.args.seed})",
                  self.is_main)

    # ---------------------------------------------------------------- model --
    def _load_pretrained(self, ckpt_path):
        data = torch.load(ckpt_path, map_location='cpu')
        state = data['model'] if isinstance(data, dict) and 'model' in data else data
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}

        own = self.model.state_dict()
        mismatched = [
            f"{k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}"
            for k, v in state.items() if k in own and v.shape != own[k].shape
        ]
        if mismatched:
            raise RuntimeError(
                "Pretrained checkpoint does not match the target architecture "
                "(img_size/img_channel/T_in/T_out/dim/wave/level/hf_mode must "
                "equal the SEVIR pixel run):\n  " + "\n  ".join(mismatched[:10]))

        missing = set(own) - set(state)
        if missing:
            raise RuntimeError(
                f"Checkpoint is missing {len(missing)} keys the model expects - "
                f"wrong --backbone? (DAWNCast for dawncast.py naming, "
                f"DAWNCast_old for dawncast_old.py):\n  {sorted(missing)[:5]}")

        self.model.load_state_dict(state, strict=True)
        print_log(f"Loaded pretrained weights from {ckpt_path} "
                  f"(epoch {data.get('epoch')}, step {data.get('step')})", self.is_main)

    def _build_model(self):
        super()._build_model()

        self._load_pretrained(self.ft_args.pretrained_ckpt)

        sel, union = apply_unfreeze(self.model, self.ft_args.unfreeze)
        _, _, _, self.trainable_names = audit_trainable(
            self.model, sel, union, self.is_main, zero_shot=self.ft_args.zero_shot)

        locked = lock_frozen_norms(self.model)
        print_log(f"Locked {len(locked)} frozen norm layers into eval()", self.is_main)

        # EMA was built by the base class over the randomly initialised weights.
        self.ema = EMA(self.model, beta=self.args.ema_rate, update_every=20).to(self.device)

        self.frozen_snapshot = snapshot_frozen(self.model)
        print_log(f"Snapshotted {len(self.frozen_snapshot)} frozen tensors for the "
                  f"step-{self.ft_args.freeze_check_step} sanity check", self.is_main)

    # ------------------------------------------------------------ optimizer --
    def _build_optimizer(self):
        # Base builds AdamW over filter(lambda p: p.requires_grad, model.parameters()).
        super()._build_optimizer()

        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        assert all(p.requires_grad for p in opt_params), "frozen parameter reached the optimizer"
        opt_numel = sum(p.numel() for p in opt_params)
        trainable_numel = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        assert opt_numel == trainable_numel, (
            f"optimizer covers {opt_numel:,} params but {trainable_numel:,} are trainable")
        print_log(f"Optimizer over {len(opt_params)} tensors / {opt_numel:,} params", self.is_main)

    # ---------------------------------------------------------------- checks --
    def _gamma_min(self):
        model = self.accelerator.unwrap_model(self.model)
        vals = [p.detach().min().item() for n, p in model.named_parameters()
                if n.endswith('gabor.gamma')]
        return min(vals) if vals else None

    def _assert_frozen_unchanged(self):
        model = self.accelerator.unwrap_model(self.model)
        moved = []
        for name, p in model.named_parameters():
            ref = self.frozen_snapshot.get(name)
            if ref is None:
                continue
            cur = p.detach().cpu()
            if not torch.equal(cur, ref):
                moved.append((name, (cur.float() - ref.float()).abs().max().item()))

        if moved:
            detail = "\n  ".join(f"{n}  max|delta|={d:.3e}" for n, d in moved[:20])
            raise RuntimeError(
                f"FROZEN PARAMETERS MOVED after {self.cur_step} steps "
                f"({len(moved)} tensors changed):\n  {detail}")
        print_log(f" ========= Freeze check passed at step {self.cur_step}: "
                  f"{len(self.frozen_snapshot)} frozen tensors bit-identical ==========",
                  self.is_main)

    def _train_batch(self, batch):
        if not self.freeze_check_done and self.cur_step >= self.ft_args.freeze_check_step:
            self._assert_frozen_unchanged()
            self.freeze_check_done = True

        # gamma is a trainable, unconstrained Parameter entering as
        # exp(-0.5 * D * gamma): negative gamma expands the envelope instead of
        # decaying it. The pixel runner's train loop has no epoch hook, so poll.
        if self.cur_step % 200 == 0:
            gmin = self._gamma_min()
            if gmin is not None:
                if self.is_main:
                    self.accelerator.log({'gabor/gamma_min': gmin}, step=self.cur_step)
                if gmin <= 0:
                    print_log(f"WARNING: Gabor gamma went non-positive (min {gmin:.4e}) "
                              f"at step {self.cur_step} - expect divergence", self.is_main)

        return super()._train_batch(batch)

    # ---------------------------------------------------------------- saving --
    def save(self, svname=None):
        super().save(svname)
        if not self.is_main:
            return

        model = self.accelerator.unwrap_model(self.model)
        adapter = {name: p.detach().cpu().clone()
                   for name, p in model.named_parameters() if p.requires_grad}
        tag = self.ft_args.target_tag
        stem = svname if svname is not None else self.cur_step
        path = osp.join(self.ckpt_path, f"adapter-{tag}-{stem}.pt")
        torch.save({
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'space': 'pixel',
            'target_dataset': self.args.dataset,
            'target_tag': tag,
            'pretrained_ckpt': self.ft_args.pretrained_ckpt,
            'unfreeze': self.ft_args.unfreeze,
            'trainable_param_names': list(adapter.keys()),
            'adapter': adapter,
        }, path)
        print_log(f"Save {tag} adapter ({sum(v.numel() for v in adapter.values()):,} params) "
                  f"to {path}", self.is_main)


# =============================================================================
# Args
# =============================================================================

def parse_transfer_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pretrained_ckpt",   type=str,   default=PRETRAINED_PIXEL_CKPT,
                        help="pretrained SEVIR pixel-space checkpoint to transfer from")
    parser.add_argument("--unfreeze",          type=str,   nargs='+', default=['temporal'],
                        choices=['temporal', 'lifting', 'projection', 'norms', 'biases',
                                 'norms_stem', 'biases_stem', 'dw_spatial'],
                        help="adaptation surface(s) to unfreeze; everything else stays frozen")
    parser.add_argument("--zero_shot",         action="store_true",
                        help="evaluate the pretrained checkpoint on the target test set, no training")
    parser.add_argument("--train_frac",        type=float, default=1.0,
                        help="fraction of the training set to fine-tune on (valid/test stay full)")
    parser.add_argument("--target_tag",        type=str,   default=None,
                        help="tag for the lightweight adapter checkpoint (default: dataset name)")
    parser.add_argument("--freeze_check_step", type=int,   default=10,
                        help="step at which frozen parameters are asserted unchanged")
    parser.add_argument("--results_csv",       type=str,
                        default="/home/vatsal/Dataserver2/Neurips/csv_files/Transfer_runs_pixel.csv",
                        help="CSV path for the shared results logger")
    parser.add_argument("--csv_log_val",       action="store_true",
                        help="also write a CSV row for validation passes")
    ft_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return ft_args


def main():
    ft_args = parse_transfer_args()
    args = create_parser()

    if args.gpu_use:
        gpu_list = ','.join(args.gpu_use)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
        print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    if ft_args.target_tag is None:
        ft_args.target_tag = args.dataset

    surface = "zero-shot (no training)" if ft_args.zero_shot else '+'.join(ft_args.unfreeze)
    patch_results_logger(
        ft_args.results_csv,
        transfer_why=f"PIXEL SEVIR->{args.dataset} transfer | unfrozen: {surface} | lr {args.lr}",
    )

    exp = PixelTransferRunner(args, ft_args)

    if ft_args.zero_shot:
        # No training: weights are already loaded, so go straight to the target
        # test set. args.eval is flipped only now - setting it before
        # construction would make _preparation look for a ckpt-best.pt that
        # does not exist yet.
        exp.args.eval = True
        exp.test_samples('zeroshot', do_test=True)
        print_log("Zero-shot evaluation done", exp.is_main)
    elif not args.eval:
        exp.train()
    else:
        exp.check_milestones(target_ckpt=args.ckpt_milestone)


if __name__ == '__main__':
    main()
