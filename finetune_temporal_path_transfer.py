"""
=============================================================================
SEMI-FOUNDATION-MODEL TRANSFER — temporal processing path only
=============================================================================

Loads a DAWN-Cast / AmpliNet checkpoint pretrained on the SEVIR latent dataset,
freezes the ENTIRE network, and re-opens ONLY the temporal processing path
inside every WaveletGaborBlock - everything above the IDWT reconstruction:

        DWT (parameter-free)
          |
          +-- BandTemporalStream.gabor    (GaborLayer)      TRAINABLE  <-- cut
          +-- BandTemporalStream.mlp      (Linear-SELU-Lin) TRAINABLE  <-- cut
          +-- BandTemporalStream.fusion   (Conv3d 1x1)      TRAINABLE  <-- cut
          |
        reconstructed = self.idwt(...) (parameter-free)     FROZEN boundary
        conv_spectral (SRST stack)                          FROZEN
        x = x_st + gabor_residual   (Gabor residual add)    FROZEN
        lifting / projection                                FROZEN

All temporal processing - including the Gabor stream - is trainable; everything
from the IDWT reconstruction downwards is frozen. For the SEVIR config
(db6, J=2, hf_mode=separate) that is 158,496 trainable of 59,543,204
parameters (0.27%) - GaborLayer.gamma is an nn.Parameter, so the Gabor
envelope width trains too.

--unfreeze selects the adaptation surface (any combination, union-ed):

    temporal    158,496   BandTemporalStream: gabor + mlp + fusion   [default]
    lifting     118,144   latent -> hidden stem
    projection  148,484   hidden -> latent head
    norms        11,264   GroupNorm affine (weight = gamma, bias = beta)
    biases       86,264   every additive vector: *.bias plus AFNO b1/b2
    dw_spatial   46,080   depthwise spatial convs in SpectralBlock_2D

--zero_shot skips training entirely and evaluates the pretrained checkpoint
on the target test set, as the transfer floor.

The runner, dataloaders, per-dataset frozen AutoencoderKL, metrics, WandB
tracker and CSV results logger are all reused as-is from
run_alphapre_convlstm_sevir_lr_latent.py - this file only subclasses Runner.
No production file is modified.

USAGE (train):
    CUDA_VISIBLE_DEVICES=0 python finetune_temporal_path_transfer.py \
        --pretrained_ckpt /path/to/ckpt-best.pt \
        --backbone DAWNCast_old \
        --dataset meteo_lr_latent_32 --img_size 32 --img_channel 4 \
        --frames_in 5 --frames_out 20 --seq_len 25 \
        --wave db6 --wavelet_level 2 --hf_mode separate \
        --gpu_use 0 --valid

USAGE (test + CSV row):
    ... same flags ... --eval
=============================================================================
"""
import os
import os.path as osp
import re
import sys
import time
import logging
import argparse

import torch
from torch import nn
from tqdm import tqdm
from ema_pytorch import EMA

from utils.tools import print_log
import utils.results_logger_csv as results_logger_csv

from run_alphapre_convlstm_sevir_lr_latent import (
    MODEL_REGISTRY,
    Runner,
    create_parser,
)

# =============================================================================
# The SEVIR checkpoint was trained with the AmpliNet module names
# (lastocast.operator.stream_ll / hf_streams / conv_spectral), which is what
# models/DAWNCast/dawncast_old.py defines. dawncast.py uses the renamed
# modules (dawncast.wgtm.fat_ll / srst) and would not load strict=True.
# Register the checkpoint-compatible module here instead of editing the runner.
# =============================================================================
MODEL_REGISTRY["DAWNCast_old"] = {
    "module": "models.DAWNCast.dawncast_old",
    "kwargs_type": "dawncast",
}

# Per-dataset frozen AutoencoderKL (used only if --ae_ckpt_path is not given).
AE_CKPT_PATH = "/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Pretrained_ae_checkpoints"
TARGET_AE_CKPT = {
    'meteo_lr_latent_32':    osp.join(AE_CKPT_PATH, "autoencoder_checkpoint_32_METEONET.pth"),
    'shanghai_lr_latent_32': osp.join(AE_CKPT_PATH, "autoencoder_checkpoint_32_SHANGHAI.pth"),
    'cikm_latent_32':        osp.join(AE_CKPT_PATH, "autoencoder_checkpoint_32_CIKM.pth"),
    'sevir_lr_latent_32':    osp.join(AE_CKPT_PATH, "autoencoder_checkpoint_32_SEVIR.pth"),
}

# Every trainable parameter must match this. The unfreeze itself is done by
# module traversal (below); this pattern is the audit that catches drift.
TRAINABLE_NAME_RE = re.compile(
    r"^lastocast\.(operator|wgtm)\."
    r"(stream_ll|stream_hf|hf_streams\.\d+|fat_ll|fat_hf|fat_hf_streams\.\d+)\."
    r"(gabor\.(mu|freq|gamma|linear\.(weight|bias))|mlp\.\d+\.(weight|bias)|fusion\.(weight|bias))$"
)

NORM_TYPES = (
    nn.GroupNorm,
    nn.LayerNorm,
    nn.modules.batchnorm._BatchNorm,
    nn.modules.instancenorm._InstanceNorm,
)


# =============================================================================
# Freezing helpers
# =============================================================================

def collect_temporal_path_modules(model):
    """
    Return {qualified_name: module} for the temporal processing slice: every
    BandTemporalStream (FAT Block) inside every WaveletGaborBlock, in full -
    gabor + mlp + fusion. That is everything above the IDWT reconstruction;
    conv_spectral, the Gabor residual merge and everything downstream are
    deliberately excluded.
    """
    targets = {}
    for blk_name, blk in model.named_modules():
        if type(blk).__name__ not in ('WaveletGaborBlock', 'WGTMBlock'):
            continue

        for child_name, child in blk.named_children():
            if type(child).__name__ in ('BandTemporalStream', 'FATBlock'):
                targets[f"{blk_name}.{child_name}"] = child
            elif isinstance(child, nn.ModuleList):          # hf_streams / fat_hf_streams
                for i, sub in enumerate(child):
                    if type(sub).__name__ in ('BandTemporalStream', 'FATBlock'):
                        targets[f"{blk_name}.{child_name}.{i}"] = sub

    if not targets:
        raise RuntimeError(
            "No WaveletGaborBlock/BandTemporalStream found - is --backbone a DAWNCast variant?"
        )
    return targets


def collect_unfreeze_groups(model, groups):
    """
    Return {group_name: {param_name: param}} for each requested adaptation
    surface. Groups may overlap (e.g. 'biases' includes norm.bias); the caller
    unions them. Selection is by module traversal / parameter kind, never by
    hand-written name lists.

      temporal   - every BandTemporalStream (gabor + mlp + fusion)
      lifting    - the latent -> hidden stem
      projection - the hidden -> latent head
      norms      - GroupNorm affine (weight = gamma, bias = beta), whole model
      biases     - every additive vector: *.bias plus AFNO b1/b2, whole model
      norms_stem  \  same two, but scoped to the temporal streams, lifting and
      biases_stem /  projection only - conv_spectral (the frozen SRST trunk,
                     which holds >90% of both) is excluded
      dw_spatial - depthwise spatial convs inside SpectralBlock_2D
    """
    sel = {g: {} for g in groups}

    def in_spectral(name):
        return '.conv_spectral.' in name or '.srst.' in name

    if 'temporal' in sel:
        for mod_name, module in collect_temporal_path_modules(model).items():
            for pn, p in module.named_parameters(recurse=True):
                sel['temporal'][f"{mod_name}.{pn}"] = p

    for stem in ('lifting', 'projection', 'dw_spatial'):
        if stem not in sel:
            continue
        for mod_name, module in model.named_modules():
            if mod_name.split('.')[-1] != stem:
                continue
            for pn, p in module.named_parameters(recurse=True):
                sel[stem][f"{mod_name}.{pn}"] = p

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


def lock_frozen_norms(model):
    """
    Put every fully-frozen norm layer into .eval() and make its train() a no-op,
    so a later model.train() (called once per epoch by Runner.train) cannot put
    it back into training mode and let running stats drift.
    """
    locked = []
    for name, m in model.named_modules():
        if not isinstance(m, NORM_TYPES):
            continue
        if any(p.requires_grad for p in m.parameters(recurse=True)):
            continue

        m.eval()

        def _train_noop(mode=True, _m=m):
            _m.training = False
            return _m

        m.train = _train_noop
        locked.append(name)
    return locked


def snapshot_frozen(model):
    """CPU clone of every frozen parameter, for the post-training equality check."""
    return {
        name: p.detach().clone().cpu()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }


# Per-group name patterns. The unfreeze itself is done by module traversal;
# these are the independent audit that catches any drift.
GROUP_NAME_RE = {
    'temporal':   TRAINABLE_NAME_RE,
    'lifting':    re.compile(r"^lastocast\.lifting\."),
    'projection': re.compile(r"^lastocast\.projection\."),
    'norms':      re.compile(r"\.norm\.(weight|bias)$"),
    'biases':     re.compile(r"(\.bias|\.b1|\.b2)$"),
    'norms_stem': re.compile(r"^(?!.*\.conv_spectral\.).*\.norm\.(weight|bias)$"),
    'biases_stem': re.compile(r"^(?!.*\.conv_spectral\.).*(\.bias|\.b1|\.b2)$"),
    'dw_spatial': re.compile(r"\.dw_spatial\.weight$"),
}


def audit_trainable(model, sel, union, is_main=True, zero_shot=False):
    """
    Print the parameter audit and assert the trainable set is EXACTLY the
    union of the selected groups - no more, no less.
    """
    total, trainable, frozen = 0, 0, 0
    trainable_named = []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            trainable_named.append((name, tuple(p.shape), p.numel()))
        else:
            frozen += p.numel()

    header = "ZERO-SHOT (no training)" if zero_shot else "TRANSFER FINE-TUNE"
    print_log("=" * 84, is_main)
    print_log(f"  FREEZE AUDIT - {header}   groups: {'+'.join(sel.keys())}", is_main)
    print_log("=" * 84, is_main)
    print_log(f"  total params     : {total:,}", is_main)
    print_log(f"  trainable params : {trainable:,}  ({100.0 * trainable / total:.4f}%)", is_main)
    print_log(f"  frozen params    : {frozen:,}  ({100.0 * frozen / total:.4f}%)", is_main)
    for g, members in sel.items():
        n_par = sum(p.numel() for p in members.values())
        print_log(f"    group {g:<12} {len(members):>3} tensors  {n_par:,} params", is_main)
    print_log("-" * 84, is_main)
    for name, shape, numel in trainable_named:
        print_log(f"    [trainable] {name:<62} {str(shape):<18} {numel:,}", is_main)
    print_log("=" * 84, is_main)

    assert total == trainable + frozen, "parameter accounting mismatch"
    assert trainable > 0, "no trainable parameters left - freezing is misconfigured"

    # 1. trainable set == selected union, exactly (catches over- and under-freezing)
    got, want = {n for n, _, _ in trainable_named}, set(union.keys())
    if got != want:
        raise AssertionError(
            f"Trainable set does not match the selected groups.\n"
            f"  unexpected trainable ({len(got - want)}): {sorted(got - want)[:10]}\n"
            f"  expected but frozen  ({len(want - got)}): {sorted(want - got)[:10]}"
        )

    # 2. every selected name matches its group's pattern
    for g, members in sel.items():
        pat = GROUP_NAME_RE[g]
        bad = [n for n in members if not pat.search(n)]
        if bad:
            raise AssertionError(f"Group '{g}' selected off-pattern parameters: {bad[:10]}")

    # 3. nothing from a group that was NOT requested may be open
    for name, _, _ in trainable_named:
        if any(pat.search(name) for g, pat in GROUP_NAME_RE.items() if g not in sel):
            if name not in union:
                raise AssertionError(f"Parameter from an unrequested group is trainable: {name}")

    return total, trainable, frozen, [n for n, _, _ in trainable_named]


# =============================================================================
# CSV results logger - same logger, transfer-tagged, configurable path
# =============================================================================

def patch_results_logger(target_csv, transfer_why):
    """
    Runner.test_samples does `from utils.results_logger_csv import ResultsLogger`
    inside the function with a hard-coded csv_path. Rebinding the class here
    redirects the row to the transfer CSV and tags it, without editing the runner.
    """
    base_cls = results_logger_csv.ResultsLogger

    class TransferResultsLogger(base_cls):
        def __init__(self, csv_path=None):          # runner's path is overridden
            super().__init__(csv_path=target_csv)

        def log_results(self, *args, **kwargs):
            kwargs.setdefault("why", transfer_why)
            return super().log_results(*args, **kwargs)

    results_logger_csv.ResultsLogger = TransferResultsLogger


# =============================================================================
# Runner
# =============================================================================

class TransferRunner(Runner):
    """Runner with the pretrained load, the freeze, the audit and the checks."""

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

        # Deterministic subsample of the TRAINING set only; valid/test stay full
        # so the numbers remain comparable with the 100%-data runs.
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
                "(T_in/T_out/dim/wave/level/hf_mode must equal the SEVIR run):\n  "
                + "\n  ".join(mismatched[:10])
            )

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
            f"optimizer covers {opt_numel:,} params but {trainable_numel:,} are trainable"
        )
        print_log(f"Optimizer over {len(opt_params)} tensors / {opt_numel:,} params", self.is_main)

    # ----------------------------------------------------------- freeze check --
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
                f"({len(moved)} tensors changed):\n  {detail}"
            )
        print_log(f" ========= Freeze check passed at step {self.cur_step}: "
                  f"{len(self.frozen_snapshot)} frozen tensors bit-identical ==========",
                  self.is_main)

    def _train_batch(self, batch):
        if not self.freeze_check_done and self.cur_step >= self.ft_args.freeze_check_step:
            self._assert_frozen_unchanged()
            self.freeze_check_done = True
        return super()._train_batch(batch)

    # --------------------------------------------------------------- training --
    def _gamma_min(self):
        """Smallest Gabor envelope width across all streams, or None if fixed."""
        model = self.accelerator.unwrap_model(self.model)
        vals = [p.detach().min().item() for n, p in model.named_parameters()
                if n.endswith('gabor.gamma')]
        return min(vals) if vals else None

    def train(self):
        """
        Copy of Runner.train() with the hard `if (epoch+1) == 35: break` removed,
        so the fine-tune runs the full --epochs. Everything else is unchanged;
        the base method is left alone because every other experiment shares it.
        """
        self.ae = self.load_autoencoder(self.ae_model, self.ae_ckpt, "cuda")
        start_epoch = self.cur_epoch
        for epoch in range(start_epoch, self.global_epochs):

            print(f"Training : {epoch+1}")
            epoch_start_time = time.time()
            self.cur_epoch = epoch
            self.model.train()
            epoch_loss_sum = 0.0
            epoch_steps    = 0

            for i, batch in enumerate(tqdm(self.train_loader, total=len(self.train_loader))):
                with self.accelerator.autocast(self.model):
                    loss_dict = self._train_batch(batch)
                    self.accelerator.backward(loss_dict['total_loss'])

                    if self.cur_step == 0:
                        # training process check
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                print_log(name, self.is_main)

                self.accelerator.wait_for_everyone()

                grad_norm = None
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                if not self.accelerator.optimizer_step_was_skipped:
                    self.scheduler.step()

                # record train info
                lr = self.optimizer.param_groups[0]['lr']
                log_dict = dict()
                log_dict['lr'] = lr

                if grad_norm is not None:
                    log_dict['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

                for k, v in loss_dict.items():
                    if type(v) == float:
                        log_dict[k] = v
                    else:
                        log_dict[k] = v.item()
                epoch_loss_sum += log_dict.get('total_loss', 0.0)
                epoch_steps    += 1
                self.accelerator.log(log_dict, step=self.cur_step)

                state_str = f"Epoch {self.cur_epoch}/{self.global_epochs}, Step {i}/{self.steps_per_epoch}"

                if i % 200 == 0:
                    logging.info(state_str + '::' + str(log_dict))
                self.ema.update()

                self.cur_step += 1

                # do santy check at begining
                if self.cur_step == 1:
                    """ santy check """
                    if not osp.exists(self.sanity_path):
                        try:
                            print_log(f" ========= Running Sanity Check ==========", self.is_main)
                            radar_ori, radar_recon = self._sample_batch(batch)
                            os.makedirs(self.sanity_path)
                            print_log(f" ========= Sanity Check over ==========", self.is_main)
                        except Exception as e:
                            print_log(e, self.is_main)
                            print_log("Sanity Check Failed", self.is_main)

            if self.is_main:
                epoch_avg_loss = epoch_loss_sum / max(epoch_steps, 1)
                log_epoch = {
                    'epoch/total_loss': epoch_avg_loss,
                    'epoch/index':      epoch + 1,
                }
                # GaborLayer.gamma is now a trainable, unconstrained Parameter and
                # enters as exp(-0.5 * D * gamma): a negative gamma blows the
                # envelope up instead of decaying it. Track its minimum.
                gmin = self._gamma_min()
                if gmin is not None:
                    log_epoch['gabor/gamma_min'] = gmin
                    if gmin <= 0:
                        print_log(f"WARNING: Gabor gamma went non-positive (min {gmin:.4e}) "
                                  f"at epoch {epoch+1} - envelope is now expanding, "
                                  f"expect divergence", self.is_main)
                self.accelerator.log(log_epoch, step=self.cur_step)
                print_log(f"Epoch {epoch+1} avg train loss: {epoch_avg_loss:.6f}"
                          + (f" | gamma_min {gmin:.4f}" if gmin is not None else ""), self.is_main)

            # save checkpoint and do test every 5 epochs
            if self.args.valid:

                if (epoch+1) % 5 == 0:
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

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    # ---------------------------------------------------------------- saving --
    def save(self, svname=None):
        super().save(svname)
        if not self.is_main:
            return

        model = self.accelerator.unwrap_model(self.model)
        adapter = {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad
        }
        tag = self.ft_args.target_tag
        stem = svname if svname is not None else self.cur_step
        path = osp.join(self.ckpt_path, f"adapter-{tag}-{stem}.pt")
        torch.save({
            'step': self.cur_step,
            'epoch': self.cur_epoch,
            'target_dataset': self.args.dataset,
            'target_tag': tag,
            'pretrained_ckpt': self.ft_args.pretrained_ckpt,
            'trainable_param_names': list(adapter.keys()),
            'adapter': adapter,
        }, path)
        print_log(f"Save {tag} adapter ({sum(v.numel() for v in adapter.values()):,} params) "
                  f"to {path}", self.is_main)

    # --------------------------------------------------------------- logging --
    def test_samples(self, milestone, epoch=None, do_test=False):
        # Base writes the CSV row only under --eval; open it for validation too.
        if self.ft_args.csv_log_val and self.is_main and not self.args.eval:
            was_eval = self.args.eval
            self.args.eval = True
            try:
                return super().test_samples(milestone, epoch=epoch, do_test=do_test)
            finally:
                self.args.eval = was_eval
        return super().test_samples(milestone, epoch=epoch, do_test=do_test)


# =============================================================================
# Args
# =============================================================================

def parse_transfer_args():
    """
    Consume the transfer-only flags, then hand the rest to the runner's
    create_parser() unchanged (it calls parse_args() on sys.argv itself).
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--pretrained_ckpt",   type=str,   required=True,
                        help="pretrained SEVIR latent checkpoint to transfer from")
    parser.add_argument("--unfreeze",          type=str,   nargs='+', default=['temporal'],
                        choices=['temporal', 'lifting', 'projection', 'norms', 'biases',
                                 'norms_stem', 'biases_stem', 'dw_spatial'],
                        help="adaptation surface(s) to unfreeze; everything else stays frozen")
    parser.add_argument("--train_frac",        type=float, default=1.0,
                        help="fraction of the training set to fine-tune on (valid/test stay full)")
    parser.add_argument("--zero_shot",         action="store_true",
                        help="evaluate the pretrained checkpoint on the target test set with no training")
    parser.add_argument("--target_tag",        type=str,   default=None,
                        help="tag for the lightweight adapter checkpoint (default: dataset name)")
    parser.add_argument("--freeze_check_step", type=int,   default=10,
                        help="step at which frozen parameters are asserted unchanged")
    parser.add_argument("--results_csv",       type=str,
                        default="/home/vatsal/Dataserver2/Neurips/csv_files/Transfer_runs.csv",
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

    # Per-dataset frozen AutoencoderKL, unless one was passed explicitly.
    if "--ae_ckpt_path" not in sys.argv and args.dataset in TARGET_AE_CKPT:
        args.ae_ckpt_path = TARGET_AE_CKPT[args.dataset]
        print(f"[Auto] AE checkpoint for {args.dataset}: {args.ae_ckpt_path}")

    if ft_args.target_tag is None:
        ft_args.target_tag = args.dataset

    surface = "zero-shot (no training)" if ft_args.zero_shot else '+'.join(ft_args.unfreeze)
    patch_results_logger(
        ft_args.results_csv,
        transfer_why=f"SEVIR->{args.dataset} transfer | unfrozen: {surface} | lr {args.lr}",
    )

    exp = TransferRunner(args, ft_args)

    if ft_args.zero_shot:
        # No training: the pretrained weights are already loaded, so go straight
        # to the target test set. args.eval is flipped only now - setting it
        # before construction would make _preparation look for a ckpt-best.pt
        # that does not exist yet.
        exp.args.eval = True
        exp.test_samples('zeroshot', do_test=True)
        print_log("Zero-shot evaluation done", exp.is_main)
    elif not args.eval:
        exp.train()
    else:
        exp.check_milestones(target_ckpt=args.ckpt_milestone)


if __name__ == '__main__':
    main()
