"""
Test-set evaluation for a pixel-space run, with per-threshold CSI.

Used by `select_best_init.py` (Part C gate) and for the final results tables.
Uses `PerThresholdEvaluator`, so it returns CSI-181/CSI-219 style columns as well
as the threshold-averaged metrics.
"""

import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from THE_GABOR.datasets.pixel_datasets import build_pixel_loaders
from THE_GABOR.utils.experiment import DEFAULT_OUTPUT_ROOT
from THE_GABOR.utils.metrics_per_threshold import PerThresholdEvaluator


# Per-dataset SRST geometry, from the ORIGINAL TASK. Needed because runs trained
# before the config fix did not persist these four values; falling back to the
# generic defaults silently builds a DIFFERENT network and the load fails.
DATASET_ARCH = {
    'cikm':     dict(afno_blocks=1, afno_hidden_size_factor=1, k_spatial=7),
    'meteo':    dict(afno_blocks=4, afno_hidden_size_factor=4, k_spatial=3),
    'shanghai': dict(afno_blocks=4, afno_hidden_size_factor=3, k_spatial=3),
    'sevir':    dict(afno_blocks=4, afno_hidden_size_factor=4, k_spatial=3),
}


def _arch(cfg, key):
    """Prefer the saved value; else recover it from the dataset table."""
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    ds = cfg.get('dataset')
    if ds in DATASET_ARCH and key in DATASET_ARCH[ds]:
        return DATASET_ARCH[ds][key]
    raise KeyError(f'cannot determine {key} for dataset {ds!r}')


def _build(cfg, total_steps=1):
    """Rebuild the exact model a run used, from its saved config."""
    ab = cfg.get('ablation', 'none')
    common = dict(
        T_in=cfg['frames_in'], T_out=cfg['frames_out'], img_channels=cfg['img_channel'],
        dim=cfg['hidden_dim'], afno_blocks=_arch(cfg, 'afno_blocks'),
        sparsity_threshold=cfg.get('sparsity_threshold', 0.01),
        afno_hidden_size_factor=_arch(cfg, 'afno_hidden_size_factor'),
        size_factor=cfg.get('size_factor', 1.0), total_steps=total_steps,
        const_ratio=cfg.get('facl_const_ratio', 0.1), k_spatial=_arch(cfg, 'k_spatial'),
        wave=cfg['wave'], wavelet_level=cfg['wavelet_level'], hf_mode=cfg['hf_mode'])
    ws, al, be = (cfg.get('weight_scale', 1.0), cfg.get('alpha', 1.0), cfg.get('beta', 1.0))
    fm = cfg.get('freq_multiplier', 1.0)
    if cfg.get('model') == 'GaborMLPControlled':
        from THE_GABOR.models.gabor_mlp_model import get_model as g
        return g(T_in=cfg['frames_in'], T_out=cfg['frames_out'],
                 img_channels=cfg['img_channel'], dim=cfg['hidden_dim'],
                 weight_scale=ws, alpha=al, beta=be, freq_multiplier=fm,
                 size_factor=cfg.get('size_factor', 1.0), wave=cfg['wave'],
                 wavelet_level=cfg['wavelet_level'], hf_mode=cfg['hf_mode'],
                 total_steps=total_steps, const_ratio=cfg.get('facl_const_ratio', 0.1))
    if ab and ab != 'none':
        from THE_GABOR.models.dawncast_ablations import get_model as g
        return g(ablation=ab, weight_scale=ws, alpha=al, beta=be,
                 freq_multiplier=fm, **common)
    from THE_GABOR.models.dawncast_transfer import get_model as g
    return g(weight_scale=ws, alpha=al, beta=be, freq_multiplier=fm, **common)


@torch.no_grad()
def evaluate_pixel_run(run_name, which='best', output_root=None, num_workers=8,
                       limit_batches=0, device=None):
    """Returns a dict of test_* metrics (incl. per-threshold csi_t*)."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_dir = osp.join(output_root, run_name)
    ck = torch.load(osp.join(run_dir, 'checkpoints', f'{which}_model.pt'),
                    map_location='cpu', weights_only=False)
    cfg = ck['config']

    model = _build(cfg).to(device).eval()
    model.load_state_dict(ck['model'])

    _, _, te, scale, th = build_pixel_loaders(
        cfg['dataset'], cfg['img_size'], cfg['seq_len'], cfg['stride'],
        cfg['batch_size'], num_workers, cfg['frames_in'], cfg['frames_out'])

    save_dir = osp.join(run_dir, 'test_samples'); os.makedirs(save_dir, exist_ok=True)
    ev = PerThresholdEvaluator(seq_len=cfg['frames_out'], value_scale=scale,
                               thresholds=th, save_path=save_dir)
    ti, to = cfg['frames_in'], cfg['frames_out']
    for i, batch in enumerate(tqdm(te, total=len(te), desc=f'test:{run_name}')):
        if limit_batches and i >= limit_batches:
            break
        seq = batch[:, :ti + to].to(device)
        pred, _ = model.predict(seq[:, :ti], compute_loss=False)
        ev.evaluate(seq[:, ti:].cpu().numpy(), pred.cpu().numpy())
    res = ev.done()
    del model; torch.cuda.empty_cache()
    out = {f'test_{k}': float(v) for k, v in res.items()
           if isinstance(v, (int, float, np.floating))}
    out['run'] = run_name
    out['dataset'] = cfg['dataset']
    out['ckpt_step'] = int(ck['step'])
    return out


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--which', default='best')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--limit_batches', type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(evaluate_pixel_run(a.run, a.which, num_workers=a.num_workers,
                                        limit_batches=a.limit_batches), indent=2))
