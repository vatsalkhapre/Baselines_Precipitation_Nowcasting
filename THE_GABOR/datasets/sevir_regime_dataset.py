"""
SEVIR RANDOM / STORM regime filtering (Experiment 1).

How the two regimes are represented in the SEVIR catalog
--------------------------------------------------------
Inspected `<SEVIR_ROOT>/CATALOG.csv` (and the latent copy).  Every `vil` row
belongs to exactly one of two HDF5 file families, encoded in the `file_name`
column:

    vil/2017/SEVIR_VIL_RANDOMEVENTS_2017_....h5   -> RANDOM regime
    vil/2017/SEVIR_VIL_STORMEVENTS_2017_....h5    -> STORM  regime

Counts in the pixel catalog: 16483 RANDOMEVENTS rows, 3910 STORMEVENTS rows
(20393 total, no other family).  The `event_type` column is an exact
complement of this split (NaN for all 16483 RANDOM rows, a storm label --
Thunderstorm Wind / Hail / Flash Flood / ... -- for all 3910 STORM rows), so
the two encodings agree perfectly.  `file_name` is used here because it is a
positive marker for both regimes rather than a null test.

Implementation
--------------
`datasets/dataset_sevir.py` already exposes a `catalog_filter` hook that is a
mask function applied to the whole catalog dataframe.  This module composes the
regime mask with the default `pct_missing == 0` mask and passes it through, so
neither the catalog nor the existing loader is modified in any way.

Ordering of the filters inside the existing loader is:
    start_date / end_date filter   (the train/val/test partition)
      -> catalog_filter            (pct_missing + regime, applied here)
      -> _compute_samples()        (groupby event id)
so the event-level train/val/test separation of the existing pipeline is
preserved exactly and sequences from one event never cross a split boundary.
"""

import datetime

import numpy as np
from torchvision import transforms

# Existing (unmodified) loaders
from datasets.dataset_sevir import SEVIRTorchDataset as PixelSEVIRTorchDataset
from datasets.dataset_sevir_lr_latent import SEVIRTorchDataset as LatentSEVIRTorchDataset
from datasets.get_datasets import DATAPATH

REGIMES = ('random', 'storm', 'all')

RANDOM_TAG = 'RANDOMEVENTS'
STORM_TAG = 'STORMEVENTS'

# Same partition boundaries as datasets/get_datasets.py for `sevir`
TRAIN_VALID_SPLIT = (2019, 1, 1)
VALID_TEST_SPLIT = (2019, 6, 1)
TEST_END_DATE = (2019, 12, 31)


class RegimeCatalogFilter:
    """
    Deterministic catalog mask: `pct_missing == 0` (the existing default)
    combined with the RANDOM / STORM regime selection.

    Defined as a module-level class (not a closure) so it survives pickling by
    DataLoader worker processes.
    """

    def __init__(self, regime):
        assert regime in REGIMES, f"regime must be one of {REGIMES}, got {regime}"
        self.regime = regime

    def __call__(self, catalog):
        mask = (catalog.pct_missing == 0)
        if self.regime == 'random':
            mask = mask & catalog.file_name.str.contains(RANDOM_TAG, regex=False)
        elif self.regime == 'storm':
            mask = mask & catalog.file_name.str.contains(STORM_TAG, regex=False)
        return mask

    def __repr__(self):
        return f"RegimeCatalogFilter(regime={self.regime!r})"


def pin_resize_antialias(ds, img_size):
    """
    Pin antialias=True on the 384 -> img_size resize of the pixel loader.

    torchvision changed this default between the environments used here:
        earthformer      torchvision 0.22.0 -> antialias defaults to True
        earthformer_old  torchvision 0.13.1 -> antialias defaults to None (False)
    Leaving it implicit makes the validation ground truth depend on which
    environment a run happens to use.  Measured on (8,25,384,384): switching it
    off changes the target by max 0.68 and mean 0.226 (~45% of the signal mean),
    which would shift every CSI number for reasons unrelated to the model.

    True is pinned because that is what the completed pixel runs trained and
    validated against, so results stay comparable across runs and environments.

    Applied to the constructed dataset *instance*; datasets/dataset_sevir.py is
    never modified.
    """
    ds.transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
    ])
    return ds


def _split_dates(split):
    if split == 'train':
        return None, datetime.datetime(*TRAIN_VALID_SPLIT)
    if split == 'val':
        return datetime.datetime(*TRAIN_VALID_SPLIT), datetime.datetime(*VALID_TEST_SPLIT)
    if split == 'test':
        return datetime.datetime(*VALID_TEST_SPLIT), datetime.datetime(*TEST_END_DATE)
    raise ValueError(f"unknown split {split}")


def build_sevir_regime_dataset(split, regime, img_size, seq_len, stride, batch_size,
                               latent=False, data_root=None, shuffle=None):
    """
    Build one split of SEVIR restricted to a single regime.

    All arguments other than `regime` are shared between the RANDOM and STORM
    runs, so the only difference between the two experiments is which events
    the catalog mask keeps.
    """
    assert split in ('train', 'val', 'test')
    cls = LatentSEVIRTorchDataset if latent else PixelSEVIRTorchDataset
    if data_root is None:
        data_root = DATAPATH['sevir_lr_latent_32'] if latent else DATAPATH['sevir']
    if shuffle is None:
        shuffle = (split == 'train')          # same as the existing pipeline

    start_date, end_date = _split_dates(split)
    # val/test use batch_size * 2 in the existing pipeline
    bs = batch_size if split == 'train' else batch_size * 2

    ds = cls(
        dataset_dir=data_root,
        split_mode='uneven',
        img_size=img_size,
        shuffle=shuffle,
        seq_len=seq_len,
        stride=stride,
        sample_mode='sequent',
        batch_size=bs,
        num_shard=1,
        rank=0,
        start_date=start_date,
        end_date=end_date,
        output_type=np.float32,
        preprocess=True,
        rescale_method='01',
        catalog_filter=RegimeCatalogFilter(regime),
        verbose=False,
        split=split,
    )
    if not latent:
        # latent data needs no resize; only the pixel loader has a transform
        pin_resize_antialias(ds, img_size)
    return ds


def dataset_stats(ds):
    """Number of events / sequences / batches actually used."""
    loader = ds.sevir_dataloader
    return {
        'num_events': int(loader.total_num_event),
        'num_sequences': int(loader.total_num_seq),
        'num_batches': int(len(loader)),
        'batch_size': int(loader.batch_size),
    }


def regime_sanity_report(ds, regime):
    """
    Verify -- from the filtered catalog the loader actually holds -- that only
    the requested regime survived.  Returns (ok, message).
    """
    cat = ds.sevir_dataloader.catalog
    n_random = int(cat.file_name.str.contains(RANDOM_TAG, regex=False).sum())
    n_storm = int(cat.file_name.str.contains(STORM_TAG, regex=False).sum())
    if regime == 'random':
        ok = (n_storm == 0 and n_random > 0)
    elif regime == 'storm':
        ok = (n_random == 0 and n_storm > 0)
    else:
        ok = (n_random > 0 and n_storm > 0)
    return ok, f"regime={regime}: catalog rows RANDOM={n_random} STORM={n_storm}"
