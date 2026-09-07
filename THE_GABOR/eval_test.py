"""
Test-set evaluation + Excel results table for the DAWN-Cast transfer runs.

Loads each run's `best_model.pt`, evaluates it on the SEVIR STORM latent TEST
split (decoding latents through the AE and scoring against pixel-space ground
truth, exactly as validation does), and writes one .xlsx with a row per run.

    python -m THE_GABOR.eval_test --runs A B C --out THE_GABOR/results/results.xlsx

Validation figures are read back from each run's checkpoint/W&B summary so the
table carries both val and test without re-running validation.
"""

import argparse
import json
import glob
import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from THE_GABOR.datasets.sevir_regime_dataset import build_sevir_regime_dataset, dataset_stats
from THE_GABOR.models.dawncast_transfer import get_model, subband_names
from THE_GABOR.run_latent import load_autoencoder
from THE_GABOR.utils.experiment import DEFAULT_AE_CKPT, DEFAULT_LOG_ROOT, DEFAULT_OUTPUT_ROOT


def build_from_cfg(cfg, total_steps=1):
    subs = subband_names(cfg['wavelet_level'], cfg['hf_mode'])
    return get_model(
        T_in=cfg['frames_in'], T_out=cfg['frames_out'], img_channels=cfg['img_channel'],
        dim=cfg['hidden_dim'], afno_blocks=cfg.get('afno_blocks', 4),
        sparsity_threshold=cfg.get('sparsity_threshold', 0.01),
        afno_hidden_size_factor=cfg.get('afno_hidden_size_factor', 4),
        weight_scale=cfg.get('weight_scale', 1.0), alpha=cfg.get('alpha', 1.0),
        beta=cfg.get('beta', 1.0),
        freq_multiplier=[cfg.get('freq_multiplier', 1.0)] * len(subs),
        size_factor=cfg.get('size_factor', 1.0), total_steps=total_steps,
        const_ratio=cfg.get('facl_const_ratio', 0.1), k_spatial=cfg.get('k_spatial', 3),
        wave=cfg['wave'], wavelet_level=cfg['wavelet_level'], hf_mode=cfg['hf_mode'])


def val_metrics_from_wandb(log_root, run_name):
    """Best-effort read-back of the last logged validation metrics."""
    out = {}
    for f in sorted(glob.glob(osp.join(log_root, run_name, 'wandb', 'run-*',
                                       'files', 'wandb-summary.json'))):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for k, v in d.items():
            if k.startswith('val/') and isinstance(v, (int, float)):
                out[k.replace('val/', 'val_')] = float(v)
    return out


@torch.no_grad()
def evaluate_test(run_dir, cfg, args, device):
    from utils.metrics import Evaluator
    from datasets.dataset_sevir import PIXEL_SCALE, THRESHOLDS

    model = build_from_cfg(cfg).to(device).eval()
    ck = torch.load(osp.join(run_dir, 'checkpoints', args.which + '_model.pt'),
                    map_location='cpu', weights_only=False)
    model.load_state_dict(ck['model'])

    # Evaluate on the SAME regime the model was trained on.  Reading this from
    # the run's own config (rather than hardcoding) is essential now that some
    # runs target RANDOM and some target STORM.
    regime = cfg.get('regime', 'storm')
    lat = build_sevir_regime_dataset('test', regime, img_size=cfg['img_size'],
                                     seq_len=cfg['seq_len'], stride=cfg['stride'],
                                     batch_size=cfg['batch_size'], latent=True)
    pix = build_sevir_regime_dataset('test', regime, img_size=128,
                                     seq_len=cfg['seq_len'], stride=cfg['stride'],
                                     batch_size=cfg['batch_size'], latent=False)
    st = dataset_stats(lat)
    dl_l = lat.get_torch_dataloader(num_workers=args.num_workers)
    dl_p = pix.get_torch_dataloader(num_workers=args.num_workers)

    from models.autoencoder_kl import AutoencoderKL
    ae = AutoencoderKL(in_channels=1, out_channels=1,
                       down_block_types=('DownEncoderBlock2D',) * 3,
                       up_block_types=('UpDecoderBlock2D',) * 3,
                       block_out_channels=(128, 256, 512), layers_per_block=2,
                       latent_channels=4, norm_num_groups=32)
    ae = load_autoencoder(ae, args.ae_ckpt_path, device=str(device))

    save_dir = osp.join(run_dir, 'test_samples')
    os.makedirs(save_dir, exist_ok=True)
    ev = Evaluator(seq_len=cfg['frames_out'], value_scale=PIXEL_SCALE,
                   thresholds=THRESHOLDS, save_path=save_dir)

    ti, to = cfg['frames_in'], cfg['frames_out']
    total = min(len(dl_l), len(dl_p))
    for i, (b, ob) in enumerate(tqdm(zip(dl_l, dl_p), total=total, desc='test')):
        if args.limit_batches and i >= args.limit_batches:
            break
        seq = b[:, :ti + to].to(device)
        fin = seq[:, :ti]
        std = fin.std()
        pred, _ = model.predict(fin / std, compute_loss=False)
        pred = pred * std
        B, T, C, H, W = pred.shape
        dec = ae.decode(pred.reshape(B * T, C, H, W)).view(B, T, 1, 128, 128)
        ev.evaluate(ob[:, ti:ti + to].cpu().numpy(), dec.cpu().numpy())
    res = ev.done()
    lat.sevir_dataloader.close(); pix.sevir_dataloader.close()
    del model, ae; torch.cuda.empty_cache()
    return {f'test_{k}': float(v) for k, v in res.items()
            if isinstance(v, (int, float, np.floating))}, st


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--labels', nargs='*', default=None)
    p.add_argument('--output_root', default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--log_root', default=DEFAULT_LOG_ROOT)
    p.add_argument('--ae_ckpt_path', default=DEFAULT_AE_CKPT)
    p.add_argument('--which', default='best', choices=['best', 'final', 'last'])
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--limit_batches', type=int, default=0)
    p.add_argument('--out', default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = args.labels if args.labels else args.runs
    rows = []
    for run, label in zip(args.runs, labels):
        run_dir = osp.join(args.output_root, run)
        ck = torch.load(osp.join(run_dir, 'checkpoints', args.which + '_model.pt'),
                        map_location='cpu', weights_only=False)
        cfg = ck['config']
        print(f'\n===== {run}  ({args.which}_model.pt, step={ck["step"]}) =====')
        test, st = evaluate_test(run_dir, cfg, args, device)
        val = val_metrics_from_wandb(args.log_root, run)
        init = {}
        j = osp.join(run_dir, 'initial_checkpoint.json')
        if osp.exists(j):
            init = json.load(open(j))

        row = {
            'run': run, 'label': label,
            'model': cfg.get('model'), 'space': cfg.get('space'),
            'dataset': cfg.get('dataset'), 'regime': cfg.get('regime'),
            'donor_regime': cfg.get('donor_regime'), 'donor_which': cfg.get('donor_which'),
            'donor_step': cfg.get('donor_step'),
            'transfer': cfg.get('transfer'), 'freeze': cfg.get('freeze'),
            'trainable_params': cfg.get('trainable_params'),
            'frozen_params': cfg.get('frozen_params'),
            'seed': cfg.get('seed'), 'epochs': cfg.get('epochs'),
            'steps_per_epoch': cfg.get('data/steps_per_epoch'),
            'total_steps': cfg.get('data/total_optimizer_steps'),
            'ckpt_step': int(ck['step']), 'ckpt_used': args.which,
            'wavelet': cfg.get('wavelet'), 'wavelet_level': cfg.get('wavelet_level'),
            'hf_mode': cfg.get('hf_mode'), 'hidden_dim': cfg.get('hidden_dim'),
            'frames_in': cfg.get('frames_in'), 'frames_out': cfg.get('frames_out'),
            'batch_size': cfg.get('batch_size'), 'lr': cfg.get('lr'),
            'freq_multiplier': cfg.get('freq_multiplier'),
            'FACL_only': cfg.get('FACL_only'),
            'init_sha256': init.get('sha256'),
            'test_events': st['num_events'], 'test_batches': st['num_batches'],
        }
        row.update(val); row.update(test)
        rows.append(row)
        print({k: round(v, 5) for k, v in test.items()})

    df = pd.DataFrame(rows)
    # put the headline metrics first
    head = ['label', 'donor_regime', 'transfer', 'freeze',
            'test_csi', 'test_csi4', 'test_csi16', 'test_hss',
            'test_ssim', 'test_psnr', 'test_mse', 'test_mae', 'test_rmse',
            'test_crps', 'test_lpips',
            'val_csi', 'val_hss', 'val_ssim', 'val_mse']
    cols = [c for c in head if c in df.columns] + [c for c in df.columns if c not in head]
    df = df[cols]

    out = args.out or osp.join(osp.dirname(osp.abspath(__file__)), 'results',
                               'dawncast_transfer_results.xlsx')
    os.makedirs(osp.dirname(out), exist_ok=True)
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='results', index=False)
        df[[c for c in head if c in df.columns]].to_excel(
            w, sheet_name='summary', index=False)
        ws = w.sheets['results']
        for i, c in enumerate(df.columns, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = \
                min(max(len(str(c)) + 2, 12), 34)
    df.to_csv(out.replace('.xlsx', '.csv'), index=False)
    print(f'\n[written] {out}')
    print(f'[written] {out.replace(".xlsx", ".csv")}')
    print(df[[c for c in head if c in df.columns]].to_string(index=False))


if __name__ == '__main__':
    main()
