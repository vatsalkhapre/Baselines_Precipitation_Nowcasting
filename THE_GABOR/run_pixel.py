"""
Experiment 1 -- PIXEL-space SEVIR runner.

    python -m THE_GABOR.run_pixel --regime random --seed 0
    python -m THE_GABOR.run_pixel --regime storm  --seed 0

Reference only (never modified): run_alphapre_convlstm.py
Data protocol, preprocessing, target construction and evaluation come from the
existing SEVIR pipeline; the only thing this runner adds is the RANDOM / STORM
catalog mask (see THE_GABOR/datasets/sevir_regime_dataset.py).
"""

import os
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from THE_GABOR.datasets.sevir_regime_dataset import (build_sevir_regime_dataset,
                                                     dataset_stats,
                                                     regime_sanity_report)
from THE_GABOR.utils.experiment import GaborExperiment, base_parser


class PixelGaborExperiment(GaborExperiment):
    space = 'pixel'

    def build_data(self):
        a = self.args
        from datasets.dataset_sevir import PIXEL_SCALE, THRESHOLDS
        self.pixel_scale, self.thresholds = PIXEL_SCALE, THRESHOLDS

        self.data_stats = {}
        loaders = {}
        for split in ('train', 'val', 'test'):
            ds = build_sevir_regime_dataset(
                split=split, regime=a.regime, img_size=a.img_size,
                seq_len=a.seq_len, stride=a.stride, batch_size=a.batch_size,
                latent=False, data_root=a.data_root)
            ok, msg = regime_sanity_report(ds, a.regime)
            print(f'[data] {split}: {msg}  -> {"OK" if ok else "FAILED"}')
            if not ok:
                raise RuntimeError(f'regime filtering failed for split {split}: {msg}')
            st = dataset_stats(ds)
            print(f'[data] {split}: events={st["num_events"]} '
                  f'sequences={st["num_sequences"]} batches={st["num_batches"]}')
            for k, v in st.items():
                self.data_stats[f'data/{split}_{k}'] = v
            loaders[split] = ds.get_torch_dataloader(num_workers=a.num_workers)

        self.train_loader = loaders['train']
        self.valid_loader = loaders['val']
        self.test_loader = loaders['test']
        self.steps_per_epoch = len(self.train_loader)

    def get_seq(self, batch):
        a = self.args
        seq = batch[:, :a.frames_in + a.frames_out].to(self.device)
        assert seq.shape[1] == a.frames_in + a.frames_out, 'radar sequence length error'
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
            frames_in, frames_gt = self.get_seq(batch)
            pred, _ = self.model.predict(frames_in, compute_loss=False)
            ev.evaluate(frames_gt.detach().cpu().numpy(),
                        pred.detach().cpu().numpy())
        res = ev.done()
        self.model.train()
        metric = res.get('csi', None)
        return metric, {k: v for k, v in res.items()
                        if isinstance(v, (int, float, np.floating))}


def main():
    p = base_parser('pixel')
    args = p.parse_args()
    args.frames_in, args.frames_out = 5, 20      # fixed for pixel SEVIR
    exp = PixelGaborExperiment(args)
    exp.train()


if __name__ == '__main__':
    main()
