"""
STAGE 1 (pixel) — temporal path only, for any pixel dataset.

    python -m THE_GABOR.run_stage1_pixel --dataset cikm --run_name Stage1_pixel_cikm_seed0

Trains `GaborMLPControlled` (Lifting -> DWT -> per-subband Gabor+MLP -> IDWT ->
Projection); no SRST/STR, FACL loss only, Gabor at freq_multiplier=1.0 with
freq ~ U(0,1).  Its best-validation checkpoint is the donor for Stage 2.

Generalises `run_pixel.py`, which only handles SEVIR's regime-filtered splits.
"""

import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from THE_GABOR.datasets.pixel_datasets import build_pixel_loaders, frames_for
from THE_GABOR.utils.experiment import GaborExperiment, base_parser


class Stage1PixelExperiment(GaborExperiment):
    space = 'pixel'
    model_name = 'GaborMLPControlled'

    def build_data(self):
        a = self.args
        tr, va, te, scale, th = build_pixel_loaders(
            a.dataset, a.img_size, a.seq_len, a.stride, a.batch_size,
            a.num_workers, a.frames_in, a.frames_out, a.preprocessing,
            sevir_regime=getattr(a, 'sevir_regime', 'all'))
        self.train_loader, self.valid_loader, self.test_loader = tr, va, te
        self.pixel_scale, self.thresholds = scale, th
        self.steps_per_epoch = len(tr)
        self.data_stats = {'data/train_batches': len(tr),
                           'data/val_batches': len(va),
                           'data/test_batches': len(te)}
        print(f'[data] {a.dataset}: train={len(tr)} val={len(va)} test={len(te)} batches '
              f'| scale={scale} thresholds={th}')

    def get_seq(self, batch):
        a = self.args
        seq = batch[:, :a.frames_in + a.frames_out].to(self.device)
        assert seq.shape[1] == a.frames_in + a.frames_out, \
            f'sequence length {seq.shape[1]} != {a.frames_in + a.frames_out}'
        return seq[:, :a.frames_in], seq[:, a.frames_in:]

    @torch.no_grad()
    def validate(self):
        a = self.args
        from utils.metrics_valid import Evaluator
        save_dir = osp.join(self.run_dir, 'valid_samples')
        os.makedirs(save_dir, exist_ok=True)
        ev = Evaluator(seq_len=a.frames_out, value_scale=self.pixel_scale,
                       thresholds=self.thresholds, save_path=save_dir)
        self.model.eval()
        for i, batch in enumerate(tqdm(self.valid_loader, desc='validate',
                                       total=len(self.valid_loader))):
            if a.limit_val_batches and i >= a.limit_val_batches:
                break
            fin, fgt = self.get_seq(batch)
            pred, _ = self.model.predict(fin, compute_loss=False)
            ev.evaluate(fgt.detach().cpu().numpy(), pred.detach().cpu().numpy())
        res = ev.done()
        self.model.train()
        return res.get('csi'), {k: v for k, v in res.items()
                                if isinstance(v, (int, float, np.floating))}


def main():
    p = base_parser('pixel')
    p.add_argument('--preprocessing', type=int, default=0)
    p.add_argument('--sevir_regime', type=str, default='all',
                   choices=['all', 'random', 'storm'])
    args = p.parse_args()
    fi, fo = frames_for(args.dataset)
    args.frames_in, args.frames_out = fi, fo
    args.seq_len = fi + fo
    args.img_channel = 1
    if args.run_name is None:
        args.run_name = f'Stage1_pixel_{args.dataset}_seed{args.seed}'
    Stage1PixelExperiment(args).train()


if __name__ == '__main__':
    main()
