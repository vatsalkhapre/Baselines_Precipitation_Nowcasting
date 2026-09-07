"""
Experiment 1 -- LATENT-space SEVIR runner (sevir_lr_latent_32).

    python -m THE_GABOR.run_latent --regime random --seed 0
    python -m THE_GABOR.run_latent --regime storm  --seed 0
    python -m THE_GABOR.run_latent --regime all    --seed 0

Reference only (never modified): run_alphapre_convlstm_sevir_lr_latent.py

Regime metadata in the latent pipeline
--------------------------------------
The latent dataset ships its own CATALOG.csv
(<latent_root>/CATALOG.csv) which preserves the original `file_name` column
(vil_latent/2017/SEVIR_VIL_STORMEVENTS_....h5).  Event identity and therefore
RANDOM / STORM membership IS available, so regime filtering is supported here
with exactly the same mask as in pixel space.  `--regime all` reproduces the
standard latent SEVIR experiment.

The latent data is already encoded: the autoencoder is NOT run inside the
model.  It is loaded only to decode predictions back to pixel space for
validation, which is what the existing latent runner does.
"""

import os
import os.path as osp
import sys
from collections import OrderedDict

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from THE_GABOR.datasets.sevir_regime_dataset import (build_sevir_regime_dataset,
                                                     dataset_stats,
                                                     regime_sanity_report)
from THE_GABOR.utils.experiment import DEFAULT_AE_CKPT, GaborExperiment, base_parser


def load_autoencoder(model, checkpoint_path, device='cuda'):
    """Same loading logic as run_alphapre_convlstm_sevir_lr_latent.py."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert 'model' in ckpt, "Checkpoint does not contain 'model' key"
    ckpt_model = ckpt['model']
    if isinstance(ckpt_model, dict) and all(isinstance(v, dict) for v in ckpt_model.values()):
        if len(ckpt_model) == 1:
            ckpt_state = list(ckpt_model.values())[0]
        else:
            ckpt_state = ckpt_model.get('autoencoder_kl', None)
            if ckpt_state is None:
                raise KeyError('autoencoder_kl not found in checkpoint')
    else:
        ckpt_state = ckpt_model

    new_state = OrderedDict()
    for k, v in ckpt_state.items():
        if k.startswith('module.'):
            k = k[7:]
        elif k.startswith('net.'):
            k = k[4:]
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


class LatentGaborExperiment(GaborExperiment):
    space = 'latent'

    def build_data(self):
        a = self.args
        from datasets.dataset_sevir import PIXEL_SCALE, THRESHOLDS
        self.pixel_scale, self.thresholds = PIXEL_SCALE, THRESHOLDS

        self.data_stats = {}
        latent_loaders, pixel_loaders = {}, {}
        for split in ('train', 'val', 'test'):
            ds = build_sevir_regime_dataset(
                split=split, regime=a.regime, img_size=a.img_size,
                seq_len=a.seq_len, stride=a.stride, batch_size=a.batch_size,
                latent=True, data_root=a.data_root)
            ok, msg = regime_sanity_report(ds, a.regime)
            print(f'[data] latent {split}: {msg}  -> {"OK" if ok else "FAILED"}')
            if not ok:
                raise RuntimeError(f'regime filtering failed for split {split}: {msg}')
            st = dataset_stats(ds)
            print(f'[data] latent {split}: events={st["num_events"]} '
                  f'sequences={st["num_sequences"]} batches={st["num_batches"]}')
            for k, v in st.items():
                self.data_stats[f'data/{split}_{k}'] = v
            latent_loaders[split] = ds.get_torch_dataloader(num_workers=a.num_workers)

            if split in ('val', 'test'):
                # paired pixel-space ground truth, same protocol as the
                # existing latent runner (img_size fixed to 128)
                ds_px = build_sevir_regime_dataset(
                    split=split, regime=a.regime, img_size=128,
                    seq_len=a.seq_len, stride=a.stride, batch_size=a.batch_size,
                    latent=False, data_root=None)
                st_px = dataset_stats(ds_px)
                if st_px['num_batches'] != st['num_batches']:
                    print(f'[warn] latent/pixel {split} batch counts differ '
                          f'({st["num_batches"]} vs {st_px["num_batches"]}); '
                          f'evaluation pairs the first min(...) batches.')
                pixel_loaders[split] = ds_px.get_torch_dataloader(
                    num_workers=a.num_workers)

        self.train_loader = latent_loaders['train']
        self.valid_loader = latent_loaders['val']
        self.test_loader = latent_loaders['test']
        self.valid_os_loader = pixel_loaders['val']
        self.test_os_loader = pixel_loaders['test']
        self.steps_per_epoch = len(self.train_loader)
        self.ae = None

    def get_seq(self, batch):
        a = self.args
        seq = batch[:, :a.frames_in + a.frames_out].to(self.device)
        assert seq.shape[1] == a.frames_in + a.frames_out, 'latent sequence length error'
        frames_in, frames_gt = seq[:, :a.frames_in], seq[:, a.frames_in:]
        # per-batch std normalisation, as in the existing latent runner
        std_val = frames_in.std()
        return frames_in / std_val, frames_gt / std_val

    def _get_ae(self):
        if self.ae is None:
            from models.autoencoder_kl import AutoencoderKL
            ae = AutoencoderKL(
                in_channels=1, out_channels=1,
                down_block_types=('DownEncoderBlock2D',) * 3,
                up_block_types=('UpDecoderBlock2D',) * 3,
                block_out_channels=(128, 256, 512),
                layers_per_block=2, latent_channels=4, norm_num_groups=32)
            self.ae = load_autoencoder(ae, self.args.ae_ckpt_path,
                                       device=str(self.device))
        return self.ae

    @torch.no_grad()
    def validate(self):
        a = self.args
        from utils.metrics_valid import Evaluator
        save_dir = osp.join(self.run_dir, 'valid_samples')
        os.makedirs(save_dir, exist_ok=True)
        ev = Evaluator(seq_len=a.frames_out, value_scale=self.pixel_scale,
                       thresholds=self.thresholds, save_path=save_dir)
        ae = self._get_ae()
        self.model.eval()
        total = min(len(self.valid_loader), len(self.valid_os_loader))
        for i, (batch, os_batch) in enumerate(
                tqdm(zip(self.valid_loader, self.valid_os_loader),
                     desc='validate', total=total)):
            if a.limit_val_batches and i >= a.limit_val_batches:
                break
            seq = batch[:, :a.frames_in + a.frames_out].to(self.device)
            frames_in = seq[:, :a.frames_in]
            std_val = frames_in.std()
            pred, _ = self.model.predict(frames_in / std_val, compute_loss=False)
            pred = pred * std_val

            B, T, C, H, W = pred.shape
            dec = ae.decode(pred.reshape(B * T, C, H, W))
            dec = dec.view(B, T, 1, 128, 128).detach().cpu().numpy()

            gt = os_batch[:, a.frames_in:a.frames_in + a.frames_out]
            ev.evaluate(gt.detach().cpu().numpy(), dec)
        res = ev.done()
        self.model.train()
        metric = res.get('csi', None)
        return metric, {k: v for k, v in res.items()
                        if isinstance(v, (int, float, np.floating))}


def main():
    p = base_parser('latent')
    p.add_argument('--ae_ckpt_path', type=str, default=DEFAULT_AE_CKPT,
                   help='AE checkpoint, used only to decode predictions for evaluation')
    args = p.parse_args()
    # Latent-space protocol.  The forecast horizon is deliberately identical to
    # the pixel experiment (T_in=5, T_out=20, seq_len=25) so that the pixel and
    # latent arms differ ONLY in the space they operate in.  The latent HDF5
    # files hold 49 frames per event, so seq_len=25 is fully available.
    args.img_size, args.img_channel = 32, 4
    args.frames_in, args.frames_out, args.seq_len = 5, 20, 25
    exp = LatentGaborExperiment(args)
    exp.train()


if __name__ == '__main__':
    main()
