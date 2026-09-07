"""
Generic pixel-space loaders for CIKM / MeteoNet / Shanghai / SEVIR.

`THE_GABOR/run_pixel.py` is SEVIR-only (it goes through the RANDOM/STORM catalog
mask).  Part A needs all four pixel datasets, so this wraps the repository's
existing `datasets.get_datasets.get_dataset()` and reproduces the loader
construction used by `run_alphapre_convlstm.py::_load_data` exactly:

  * SEVIR  -> `dataset.get_torch_dataloader(num_workers=...)`, batching handled
              inside the dataset object
  * others -> a normal `torch.utils.data.DataLoader`, shuffled + drop_last on train

No dataset code is modified; this only assembles what already exists.
"""

import torch

from datasets.get_datasets import get_dataset

# datasets that batch internally and expose get_torch_dataloader()
_SELF_BATCHING = ('sevir',)

# forecast horizon per dataset (matches run_alphapre_convlstm.py)
FRAMES = {'cikm': (5, 10), 'meteo': (5, 20), 'shanghai': (5, 20), 'sevir': (5, 20)}


def frames_for(dataset):
    if dataset not in FRAMES:
        raise ValueError(f'unknown pixel dataset {dataset}; known: {list(FRAMES)}')
    return FRAMES[dataset]


def build_pixel_loaders(dataset, img_size, seq_len, stride, batch_size,
                        num_workers, frames_in, frames_out, preprocessing=0,
                        sevir_regime='all'):
    """Returns (train_loader, valid_loader, test_loader, pixel_scale, thresholds).

    `sevir_regime` in {'random','storm'} restricts SEVIR to one precipitation
    regime via the catalog mask (needed by Part F). 'all' = unfiltered SEVIR.
    """
    if dataset == 'sevir' and sevir_regime in ('random', 'storm'):
        from THE_GABOR.datasets.sevir_regime_dataset import (
            build_sevir_regime_dataset, dataset_stats, regime_sanity_report)
        from datasets.dataset_sevir import PIXEL_SCALE, THRESHOLDS
        loaders = {}
        for split in ('train', 'val', 'test'):
            ds = build_sevir_regime_dataset(split, sevir_regime, img_size=img_size,
                                            seq_len=seq_len, stride=stride,
                                            batch_size=batch_size, latent=False)
            ok, msg = regime_sanity_report(ds, sevir_regime)
            if not ok:
                raise RuntimeError(f'regime filter failed on {split}: {msg}')
            st = dataset_stats(ds)
            print(f'[data] sevir/{sevir_regime} {split}: events={st["num_events"]} '
                  f'sequences={st["num_sequences"]} batches={st["num_batches"]}')
            loaders[split] = ds.get_torch_dataloader(num_workers=num_workers)
        return (loaders['train'], loaders['val'], loaders['test'],
                PIXEL_SCALE, THRESHOLDS)

    train, val, test, _color_fn, PIXEL_SCALE, THRESHOLDS = get_dataset(
        data_name=dataset, img_size=img_size, seq_len=seq_len,
        batch_size=batch_size, stride=stride, file_rain_seq_add=0, method=None,
        in_channels=frames_in, out_channels=frames_out,
        preprocess_type=preprocessing)

    if dataset in _SELF_BATCHING:
        tr = train.get_torch_dataloader(num_workers=num_workers)
        va = val.get_torch_dataloader(num_workers=num_workers)
        te = test.get_torch_dataloader(num_workers=num_workers)
    else:
        tr = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True,
                                         num_workers=num_workers, drop_last=True)
        va = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, drop_last=True)
        te = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers)
    return tr, va, te, PIXEL_SCALE, THRESHOLDS
