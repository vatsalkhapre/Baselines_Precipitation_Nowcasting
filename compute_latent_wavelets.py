"""
compute_latent_wavelet_stats.py

Rebuttal-support script for the AC comment on DAWN-Cast:
    "The physical interpretation is not sufficiently supported because
    wavelet decomposition is applied to learned latent features rather
    than directly to radar fields. It is therefore unclear whether the
    resulting subbands correspond to physically meaningful precipitation
    scales."

WHAT THIS SCRIPT DOES
----------------------
cumulative_dataset_plotting.py computes lag-1 spatial autocorrelation and
radially-averaged PSD on the *pixel-space* ground truth field and its
wavelet subbands (GT / LL / LH / HL / HH), and renders them as boxplots
and PSD curves (Fig. 1 / Fig. 5 / Appendix C of the paper).

This script performs the analogous analysis directly on the model's
*latent* representation Z = E(x) -- the actual tensor the WGTM block
wavelet-decomposes -- and reports everything as PLAIN NUMBERS (means,
stds, and a handful of scalar summaries), suitable for a text-only
OpenReview rebuttal where figures cannot be attached.

It also computes a genuinely paired pixel<->latent correspondence metric
(not just two separate analyses run side by side): for the SAME
underlying event, it correlates the pixel-space subband statistic against
the latent-space subband statistic across the sample ensemble, and it
correlates the shape of the pixel-space and latent-space PSD curves on a
common *relative* (normalized) wavenumber axis. This directly answers the
"do latent subbands correspond to physically meaningful scales" question
with a number, not just an argument.

KEY METHODOLOGICAL CHOICES (read before running)
--------------------------------------------------
1. Pixel-space frames used for the correspondence check are recomputed at
   the *same* 128x128 resolution / same crop-or-resize pipeline the
   autoencoder actually saw (replicated from each convert_*_latent.py),
   NOT the native-resolution frames used in the original Fig. 1 / Fig. 5.
   This is intentional: Fig. 1/5 characterize the *pixel field itself* at
   native resolution (correct for that purpose), but a valid pixel<->latent
   correspondence check requires comparing the latent to the exact image
   the encoder consumed. Both sets of pixel-space numbers are legitimate;
   they answer different questions. See the "PIXEL PREPROCESSING" section.

2. Wavelet level is fixed at LEVEL=1 for ALL datasets, matching the
   config already used to produce Fig. 1 / Fig. 5 / Appendix C (which
   used LEVEL=1 uniformly, distinct from the model's own per-dataset
   optimal J in Table 8). This keeps the new latent numbers directly
   comparable to the already-published pixel numbers. If you want the
   per-dataset optimal J instead, change LEVEL_DICT below and note in the
   rebuttal that levels differ from Fig. 1/5.

3. Latents have C=4 channels (from AutoencoderKL, latent_channels=4);
   pixel frames have 1 channel. Each latent channel gets its own 2D DWT,
   and channel autocorr values are POOLED across channels *and* samples
   into one ensemble (CHANNEL_AGG='pool', the default) -- this maximizes
   effective N and treats each channel as an independent spatial draw,
   which is the more information-preserving choice vs. averaging channels
   into one number per sample first (CHANNEL_AGG='mean', also computed and
   reported alongside, as a sanity check -- the two should have very
   similar central tendency).

4. PSD is reported as three SCALAR summaries per subband (not a curve),
   computed on each sample's OWN native (non-interpolated) radial grid so
   that Low/High-Frequency-Energy-Fraction values are valid fractions in
   [0, 1]:
     - LFE: fraction of power in the lowest quartile of relative wavenumber
     - HFE: fraction of power in the highest quartile
     - SLOPE: log-log linear-fit slope of the PSD (shape descriptor)
   Interpolation onto a *common* relative-wavenumber axis is used ONLY for
   the cross-space PSD-shape correlation (point 5), never for LFE/HFE/SLOPE
   themselves -- interpolating before summing does not preserve a
   normalized PSD's sum-to-1 property and would silently produce invalid
   (>1) energy fractions.

5. Cross-space PSD-shape correlation: the finer pixel-space native PSD
   curve is DOWN-sampled (via linear interp) onto the coarser latent-space
   native wavenumber grid (never the reverse -- upsampling a coarse curve
   manufactures information that is not there), then Pearson-correlated
   in log space against the latent curve.

6. SEVIR caveat: convert_sevir_latent.py only encodes year "2019" files,
   and its filename filter
       if not fname.endswith(".h5") and (fname != A or fname != B): continue
   is very likely a bug -- for any single fname, "fname != A or fname != B"
   is a tautology (a string cannot simultaneously equal both A and B, so at
   least one inequality always holds), so this condition never actually
   restricts anything: it processes every .h5 file it finds in the 2019
   directory, not just the two intended files. This script does not "fix"
   that pipeline (out of scope here); it works around it by cross-
   referencing SEVIR_CATALOG_PATH against whichever files actually exist
   in the 2019 latent directory, and restricts the paired SEVIR analysis
   to that overlap. This means the SEVIR paired sample count will likely
   be smaller than, and drawn from a different (2019-only, mixed
   storm/random) pool than, the 200-sample "SEVIR_Storm" set used in the
   original Fig. 1/5 (which draws from the full 2017-2020 catalog). Flag
   this explicitly if you report SEVIR numbers in the rebuttal, or first
   re-run convert_sevir_latent.py across all years / fix the filter so the
   pools match.

USAGE
-----
    python3 compute_latent_wavelet_stats.py                  # real data
    python3 compute_latent_wavelet_stats.py --selftest        # synthetic
                                                                check only

Requires: h5py, numpy, pandas, PyWavelets (pywt), torch, torchvision
(the last two only to replicate each dataset's exact resize/crop pipeline;
no autoencoder model or GPU is loaded -- latents are read from the
already-encoded *_latent32.h5 files produced by convert_*_latent.py).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pywt

try:
    import h5py
except ImportError:
    h5py = None  # only required for --real (non-selftest) runs

try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None
    transforms = None


# ============================================================
# CONFIGURATION
# ============================================================

WAVELET_DICT = {
    'SEVIR_Storm': 'db6',
    'CIKM': 'db6',
    'Shanghai': 'db6',
    'Meteonet': 'db6',
}
LEVEL_DICT = {k: 1 for k in WAVELET_DICT}   # LEVEL=1 for all -- see note 2 above

STRIDE = 10
NUM_SAMPLES = 200
CHANNEL_AGG_MODES = ('pool', 'mean')        # both computed; 'pool' is primary

# ---- raw pixel-space paths ----
# NOTE: cumulative_dataset_plotting.py hardcoded "Dataserver2" for these two
# paths. Confirmed with the user that the real catalog lives under
# "Dataserver" (no "2") -- matching convert_sevir_latent.py's SRC_ROOT, which
# also has no "2". Using the confirmed path here; double-check which mirror
# cumulative_dataset_plotting.py actually read from if its Fig.1/5 numbers
# need to be reproduced from this same root.
SEVIR_CATALOG_PATH = '/home/vatsal/Dataserver/Datasets/sevir/CATALOG.csv'
SEVIR_DATA_DIR = '/home/vatsal/Dataserver/Datasets/sevir/data/'
SHANGHAI_H5 = '/home/vatsal/NWM/Dataset/Shanghai_Radar/shanghai.h5'
CIKM_H5 = '/home/vatsal/NWM/Dataset/CIKM/cikm.h5'
METEONET_H5 = '/home/vatsal/NWM/Dataset/Meteonet/meteo_radar.h5'

# ---- latent-space paths (outputs of convert_*_latent.py) ----
CIKM_LATENT_H5 = '/home/vatsal/NWM/Dataset/cikm_latent_32/cikm_latent32.h5'
METEONET_LATENT_H5 = ('/home/vatsal/NWM/Dataset/meteonet_latent_32/meteonet_latent32.h5')
SHANGHAI_LATENT_H5 = '/home/vatsal/NWM/Dataset/shanghai_latent_32/shanghai_latent_data.h5'
SEVIR_LATENT_DIR = ('/home/vatsal/NWM/Dataset/sevir_lr_latent_32_normalize_resize/data/vil_latent/2019')

OUT_CSV = 'latent_wavelet_stats.csv'
OUT_SUMMARY_TXT = 'latent_wavelet_stats_rebuttal_summary.txt'

SUBBANDS = ('GT', 'LL', 'LH', 'HL', 'HH')


# ============================================================
# METRICS -- identical definitions to cumulative_dataset_plotting.py,
# plus the additional scalar-PSD-summary and correlation helpers.
# ============================================================

def calc_spatial_autocorr(img):
    if np.std(img) == 0:
        return 0.0
    h_corr = np.corrcoef(img[:, :-1].flatten(), img[:, 1:].flatten())[0, 1]
    v_corr = np.corrcoef(img[:-1, :].flatten(), img[1:, :].flatten())[0, 1]
    return (h_corr + v_corr) / 2.0


def get_radially_averaged_psd(image2d):
    f = np.fft.fft2(image2d)
    fshift = np.fft.fftshift(f)
    psd2D = np.abs(fshift) ** 2
    h, w = psd2D.shape
    y, x = np.indices((h, w))
    center = (h // 2, w // 2)
    r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2).astype(np.int32)
    tbin = np.bincount(r.ravel(), psd2D.ravel())
    nr = np.bincount(r.ravel())
    nr[nr == 0] = 1
    return tbin / nr


def normalized_relative_psd(image2d):
    """Radially-averaged PSD normalized to sum=1, indexed by k/k_max in
    [0, 1] so pixel (large grid) and latent (small grid) curves can later
    be placed on a common relative-frequency axis."""
    psd = get_radially_averaged_psd(image2d)
    total = psd.sum()
    psd = psd / total if total > 0 else psd
    k_max = len(psd) - 1
    k_rel = np.arange(len(psd)) / k_max if k_max > 0 else np.zeros_like(psd)
    return k_rel, psd


def band_energy_fractions(k_rel, psd, low=0.25, high=0.75):
    """MUST be called on a native (non-interpolated) normalized PSD, or the
    result is not a valid fraction. See module docstring, point 4."""
    lfe = psd[k_rel <= low].sum()
    hfe = psd[k_rel >= high].sum()
    return float(lfe), float(hfe)


def spectral_slope(k_rel, psd, k_min=0.05, k_max=0.95):
    mask = (k_rel >= k_min) & (k_rel <= k_max) & (psd > 0)
    if mask.sum() < 3:
        return np.nan
    logk = np.log(k_rel[mask])
    logp = np.log(psd[mask])
    slope, _ = np.polyfit(logk, logp, 1)
    return float(slope)


def interp_psd(k_rel_src, psd_src, k_rel_ref):
    return np.interp(k_rel_ref, k_rel_src, psd_src)


def pearson_r(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================
# WAVELET DECOMPOSITION
# ============================================================

def decompose_2d(img2d, wavelet, level):
    coeffs = pywt.wavedec2(img2d, wavelet, level=level)
    LL = coeffs[0]
    LH, HL, HH = coeffs[1]   # level=1 convention, matching cumulative_dataset_plotting.py
    return {'GT': img2d, 'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}


def decompose_latent_chw(latent_chw, wavelet, level):
    """Apply the 2D DWT independently to each channel of a (C,H,W) latent
    frame -- matches how the WGTM block treats the multi-channel latent as
    a stack of independent 2D spatial fields."""
    C = latent_chw.shape[0]
    out = {k: [] for k in SUBBANDS}
    for c in range(C):
        d = decompose_2d(latent_chw[c], wavelet, level)
        for k in out:
            out[k].append(d[k])
    return out


# ============================================================
# PER-SAMPLE STAT EXTRACTION
# ============================================================

def pixel_sample_stats(img2d, wavelet, level):
    subs = decompose_2d(img2d, wavelet, level)
    out = {}
    for name, arr in subs.items():
        k_rel, psd = normalized_relative_psd(arr)
        lfe, hfe = band_energy_fractions(k_rel, psd)
        out[name] = {
            'autocorr': calc_spatial_autocorr(arr),
            'k_rel': k_rel, 'psd': psd,
            'lfe': lfe, 'hfe': hfe,
            'slope': spectral_slope(k_rel, psd),
        }
    return out


def latent_sample_stats(latent_chw, wavelet, level):
    """Returns, per subband: per-channel autocorr/lfe/hfe/slope lists (for
    'pool' aggregation), their per-sample channel-mean (for 'mean'
    aggregation), and the channel-averaged native PSD curve."""
    subs = decompose_latent_chw(latent_chw, wavelet, level)
    out = {}
    for name, arr_list in subs.items():
        ac_c, lfe_c, hfe_c, slope_c, psd_c = [], [], [], [], []
        k_rel_ref = None
        for arr in arr_list:
            ac_c.append(calc_spatial_autocorr(arr))
            k_rel, psd = normalized_relative_psd(arr)
            if k_rel_ref is None:
                k_rel_ref = k_rel
            lfe, hfe = band_energy_fractions(k_rel, psd)
            lfe_c.append(lfe); hfe_c.append(hfe)
            slope_c.append(spectral_slope(k_rel, psd))
            psd_c.append(psd)
        out[name] = {
            'autocorr_per_channel': ac_c,
            'lfe_per_channel': lfe_c, 'hfe_per_channel': hfe_c,
            'slope_per_channel': slope_c,
            'autocorr_mean': float(np.mean(ac_c)),
            'lfe_mean': float(np.mean(lfe_c)), 'hfe_mean': float(np.mean(hfe_c)),
            'slope_mean': float(np.nanmean(slope_c)),
            'k_rel': k_rel_ref, 'psd_channel_mean': np.mean(psd_c, axis=0),
        }
    return out


# ============================================================
# PIXEL PREPROCESSING -- replicate each convert_*_latent.py transform
# EXACTLY, so the "pixel" side of the correspondence check is computed on
# the identical 128x128 image the encoder actually saw (see docstring,
# point 1). Requires torch/torchvision (already a dependency of the
# convert_*_latent.py scripts, so assumed present).
# ============================================================

def _require_torch():
    if torch is None or transforms is None:
        raise RuntimeError(
            "torch/torchvision are required to replicate the exact "
            "pixel preprocessing pipeline (CenterCrop/Resize) used before "
            "encoding. Install them or adapt _preprocess_* below to a "
            "pure-numpy equivalent if you cannot install torch here.")


def preprocess_cikm(frame_hw):
    """Matches convert_cikm_latent.py: CenterCrop(128) [implicit pad since
    101<128], then /255.0. (Normalization does not affect autocorr/PSD
    shape, but is applied for exact fidelity to the encoder's input.)"""
    _require_torch()
    x = torch.from_numpy(frame_hw.astype(np.float32))
    x = transforms.CenterCrop((128, 128))(x.unsqueeze(0)).squeeze(0)
    x = x / 255.0
    return x.numpy()


def preprocess_meteonet(frame_hw):
    """Matches convert_meteonet_latent.py: /90.0 FIRST, then Resize(128)."""
    _require_torch()
    x = torch.from_numpy(frame_hw.astype(np.float32)) / 90.0
    x = transforms.Resize((128, 128))(x.unsqueeze(0)).squeeze(0)
    return x.numpy()


def preprocess_shanghai(frame_hw):
    """Matches convert_shanghai_latent.py: /255.0 then Resize(128)."""
    _require_torch()
    x = torch.from_numpy(frame_hw.astype(np.float32)) / 255.0
    x = transforms.Resize((128, 128))(x.unsqueeze(0)).squeeze(0)
    return x.numpy()


def preprocess_sevir(frame_hw):
    """Matches convert_sevir_latent.py: /255.0 then Resize(128, BILINEAR)."""
    _require_torch()
    x = torch.from_numpy(frame_hw.astype(np.float32)) / 255.0
    resize = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.BILINEAR)
    x = resize(x.unsqueeze(0)).squeeze(0)
    return x.numpy()


# ============================================================
# PAIRED LOADERS -- yield (key, pixel_frame_128x128, latent_frame_CHW)
# ============================================================

def load_pairs_grouped_h5(pixel_h5_path, latent_h5_path, is_cikm,
                           preprocess_fn, max_samples=NUM_SAMPLES, stride=STRIDE):
    """For CIKM / Shanghai / Meteonet: convert_*_latent.py writes the
    latent H5 with the SAME per-sample keys as the raw H5 (verified by
    inspection of all three convert_*.py scripts, which iterate
    `source_f[split_type].keys()` and re-use `key` unchanged when writing
    the output group) -- so pairing by key is exact by construction, no
    ordinal-position assumptions needed."""
    pairs = []
    with h5py.File(pixel_h5_path, 'r') as fp, h5py.File(latent_h5_path, 'r') as fl:
        split = 'train'
        if is_cikm:
            keys = [k for k in fp[split].keys() if k.startswith('sample_')]
        else:
            if 'all_len' in fp[split]:
                n = int(fp[split]['all_len'][()])
            else:
                n = len(fp[split].keys())
            keys = [str(i) for i in range(n)]

        missing = 0
        for key in keys[::stride]:
            if len(pairs) >= max_samples:
                break
            if key not in fl[split]:
                missing += 1
                continue
            seq_pix = fp[split][key][()]          # (T, H, W)
            seq_lat = fl[split][key][()]           # (T, C, h, w)
            if seq_pix.shape[0] != seq_lat.shape[0]:
                continue  # T mismatch -- skip rather than silently misalign
            t_mid = seq_pix.shape[0] // 2
            frame_pix_raw = seq_pix[t_mid]
            frame_lat = seq_lat[t_mid]
            frame_pix = preprocess_fn(frame_pix_raw)
            pairs.append((key, frame_pix, frame_lat))
        if missing:
            print(f"  [warn] {missing} keys present in pixel H5 but missing "
                  f"from latent H5 -- skipped.")
    return pairs


def load_pairs_sevir(catalog_path, data_dir, latent_dir, event_subset='storm',
                      max_samples=NUM_SAMPLES, stride=STRIDE):
    """SEVIR pairing, restricted to the overlap between the catalog and
    whatever files actually exist in the 2019 latent directory (see
    docstring point 6 for why this restriction is necessary)."""
    if not os.path.isdir(latent_dir):
        print(f"  [warn] SEVIR latent dir not found: {latent_dir} -- skipping SEVIR.")
        return []

    # CATALOG.csv's file_name is a path relative to SEVIR_DATA_DIR, e.g.
    # "vil/2019/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5" -- but convert_sevir_latent.py
    # writes latents directly under the flat year folder using only the bare
    # filename (dst_file = os.path.join(dst_year, fname)), so latent_dir's
    # listing has NO "vil/<year>/" prefix. Comparing/joining on the bare
    # basename on both sides is required, or this silently matches nothing.
    latent_files_available = set(os.listdir(latent_dir))
    catalog = pd.read_csv(catalog_path, low_memory=False)
    vil_catalog = catalog[catalog['img_type'] == 'vil'].copy()
    vil_catalog['file_basename'] = vil_catalog['file_name'].apply(os.path.basename)
    catalog_basenames = set(vil_catalog['file_basename'].unique())
    overlap = catalog_basenames & latent_files_available
    vil_catalog = vil_catalog[vil_catalog['file_basename'].isin(latent_files_available)]
    if event_subset == 'storm':
        vil_catalog = vil_catalog[vil_catalog['event_type'].notna()]
    elif event_subset == 'random':
        vil_catalog = vil_catalog[vil_catalog['event_type'].isna()]

    if len(vil_catalog) == 0:
        print(f"  [warn] No overlap between CATALOG.csv and the available "
              f"2019 latent files.\n"
              f"    latent_dir checked:      {latent_dir}\n"
              f"    files found there ({len(latent_files_available)}): "
              f"{sorted(latent_files_available)[:10]}"
              f"{' ...' if len(latent_files_available) > 10 else ''}\n"
              f"    2019 catalog basenames ({len(catalog_basenames)}): "
              f"{sorted(catalog_basenames)[:10]}"
              f"{' ...' if len(catalog_basenames) > 10 else ''}\n"
              f"    basename overlap: {len(overlap)}\n"
              f"  If 'files found there' is 0: convert_sevir_latent.py hasn't been "
              f"(successfully) run against this exact SEVIR_LATENT_DIR, or the path "
              f"is wrong -- check DST_ROOT in convert_sevir_latent.py against "
              f"SEVIR_LATENT_DIR at the top of this script; they must resolve to the "
              f"same directory.\n"
              f"  If both sides are non-empty but overlap is 0: compare one printed "
              f"filename from each list character-by-character -- likely cause is a "
              f"naming-convention mismatch (e.g. a renamed/copied file, extra "
              f"suffix, or the catalog references filenames from a different SEVIR "
              f"release than the one actually encoded).")
        return []

    pairs = []
    for file_name in vil_catalog['file_name'].unique():
        if len(pairs) >= max_samples:
            break
        pix_path = os.path.join(data_dir, file_name)               # keep vil/<year>/ prefix -- matches raw data layout
        lat_path = os.path.join(latent_dir, os.path.basename(file_name))  # bare name -- latent_dir IS the year folder
        if not (os.path.exists(pix_path) and os.path.exists(lat_path)):
            print(f"  [warn] expected pair not found on disk: {pix_path} / {lat_path}")
            continue
        idxs = vil_catalog[vil_catalog['file_name'] == file_name]['file_index'].values
        with h5py.File(pix_path, 'r') as fp, h5py.File(lat_path, 'r') as fl:
            if 'vil_latent' not in fl:
                continue
            for idx in idxs[::stride]:
                if len(pairs) >= max_samples:
                    break
                try:
                    seq_pix = fp['vil'][idx]        # (H, W, T)
                    seq_lat = fl['vil_latent'][idx]  # (T, C, h, w)
                    t_mid_pix = seq_pix.shape[-1] // 2
                    t_mid_lat = seq_lat.shape[0] // 2
                    frame_pix_raw = seq_pix[:, :, t_mid_pix]
                    frame_lat = seq_lat[t_mid_lat]
                    frame_pix = preprocess_sevir(frame_pix_raw)
                    pairs.append((f"{file_name}:{idx}", frame_pix, frame_lat))
                except Exception as e:
                    print(f"  [warn] skipping {file_name}:{idx} ({e})")
    return pairs


DATASET_LOADERS = {
    'CIKM': lambda: load_pairs_grouped_h5(
        CIKM_H5, CIKM_LATENT_H5, is_cikm=True, preprocess_fn=preprocess_cikm),
    'Meteonet': lambda: load_pairs_grouped_h5(
        METEONET_H5, METEONET_LATENT_H5, is_cikm=False, preprocess_fn=preprocess_meteonet),
    'Shanghai': lambda: load_pairs_grouped_h5(
        SHANGHAI_H5, SHANGHAI_LATENT_H5, is_cikm=False, preprocess_fn=preprocess_shanghai),
    'SEVIR_Storm': lambda: load_pairs_sevir(
        SEVIR_CATALOG_PATH, SEVIR_DATA_DIR, SEVIR_LATENT_DIR, event_subset='storm'),
}


# ============================================================
# ENSEMBLE AGGREGATION
# ============================================================

def analyze_dataset(dataset_name, pairs, wavelet, level):
    rows = []
    # per-subband accumulators
    pix_ac = {s: [] for s in SUBBANDS}
    pix_lfe = {s: [] for s in SUBBANDS}
    pix_hfe = {s: [] for s in SUBBANDS}
    pix_slope = {s: [] for s in SUBBANDS}
    pix_native_curves = {s: [] for s in SUBBANDS}

    lat_ac_pool = {s: [] for s in SUBBANDS}
    lat_ac_mean = {s: [] for s in SUBBANDS}
    lat_lfe_pool = {s: [] for s in SUBBANDS}
    lat_hfe_pool = {s: [] for s in SUBBANDS}
    lat_slope_pool = {s: [] for s in SUBBANDS}
    lat_native_curves = {s: [] for s in SUBBANDS}

    paired_ac_pixel = {s: [] for s in SUBBANDS}
    paired_ac_latent_mean = {s: [] for s in SUBBANDS}

    for key, frame_pix, frame_lat in pairs:
        ps = pixel_sample_stats(frame_pix, wavelet, level)
        ls = latent_sample_stats(frame_lat, wavelet, level)
        for s in SUBBANDS:
            pix_ac[s].append(ps[s]['autocorr'])
            pix_lfe[s].append(ps[s]['lfe'])
            pix_hfe[s].append(ps[s]['hfe'])
            pix_slope[s].append(ps[s]['slope'])
            pix_native_curves[s].append((ps[s]['k_rel'], ps[s]['psd']))

            lat_ac_pool[s].extend(ls[s]['autocorr_per_channel'])
            lat_ac_mean[s].append(ls[s]['autocorr_mean'])
            lat_lfe_pool[s].extend(ls[s]['lfe_per_channel'])
            lat_hfe_pool[s].extend(ls[s]['hfe_per_channel'])
            lat_slope_pool[s].extend(ls[s]['slope_per_channel'])
            lat_native_curves[s].append((ls[s]['k_rel'], ls[s]['psd_channel_mean']))

            paired_ac_pixel[s].append(ps[s]['autocorr'])
            paired_ac_latent_mean[s].append(ls[s]['autocorr_mean'])

    n = len(pairs)
    for s in SUBBANDS:
        if n == 0:
            continue
        acp = np.array(pix_ac[s])
        acl_pool = np.array(lat_ac_pool[s])
        acl_mean = np.array(lat_ac_mean[s])
        r_paired_ac = pearson_r(paired_ac_pixel[s], paired_ac_latent_mean[s])

        mean_pix_curve = np.mean([c[1] for c in pix_native_curves[s]], axis=0)
        pix_k_native = pix_native_curves[s][0][0]
        mean_lat_curve = np.mean([c[1] for c in lat_native_curves[s]], axis=0)
        lat_k_native = lat_native_curves[s][0][0]
        pix_curve_on_lat_grid = interp_psd(pix_k_native, mean_pix_curve, lat_k_native)
        r_psd_shape = pearson_r(np.log(pix_curve_on_lat_grid + 1e-12),
                                 np.log(mean_lat_curve + 1e-12))

        rows.append({
            'dataset': dataset_name, 'subband': s, 'n_paired_samples': n,
            'AC_pixel_mean': acp.mean(), 'AC_pixel_std': acp.std(),
            'AC_latent_pool_mean': acl_pool.mean(), 'AC_latent_pool_std': acl_pool.std(),
            'AC_latent_meanC_mean': acl_mean.mean(), 'AC_latent_meanC_std': acl_mean.std(),
            'AC_pixel_latent_paired_r': r_paired_ac,
            'LFE_pixel': np.mean(pix_lfe[s]), 'LFE_latent_pool': np.mean(lat_lfe_pool[s]),
            'HFE_pixel': np.mean(pix_hfe[s]), 'HFE_latent_pool': np.mean(lat_hfe_pool[s]),
            'slope_pixel': np.nanmean(pix_slope[s]), 'slope_latent_pool': np.nanmean(lat_slope_pool[s]),
            'PSD_shape_corr_r': r_psd_shape,
        })
    return rows


# ============================================================
# MAIN
# ============================================================

def print_rebuttal_table(df):
    fmt_cols = ['AC_pixel_mean', 'AC_pixel_std', 'AC_latent_pool_mean',
                'AC_latent_pool_std', 'AC_pixel_latent_paired_r',
                'LFE_pixel', 'LFE_latent_pool', 'HFE_pixel', 'HFE_latent_pool',
                'slope_pixel', 'slope_latent_pool', 'PSD_shape_corr_r']
    disp = df.copy()
    for c in fmt_cols:
        disp[c] = disp[c].map(lambda v: f"{v:+.3f}" if pd.notna(v) else "nan")
    print(disp.to_string(index=False))


def write_rebuttal_summary(df, path):
    lines = []
    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        lines.append(f"\n[{dataset}]  (n={int(sub['n_paired_samples'].iloc[0])} paired samples)")
        for _, row in sub.iterrows():
            lines.append(
                f"  {row['subband']:>3s}: rho_pixel={row['AC_pixel_mean']:+.3f}"
                f"  rho_latent={row['AC_latent_pool_mean']:+.3f}"
                f"  (paired r={row['AC_pixel_latent_paired_r']:+.3f})"
                f"   LFE(pix/lat)={row['LFE_pixel']:.3f}/{row['LFE_latent_pool']:.3f}"
                f"   HFE(pix/lat)={row['HFE_pixel']:.3f}/{row['HFE_latent_pool']:.3f}"
                f"   PSD-shape r={row['PSD_shape_corr_r']:+.3f}"
            )
    text = "\n".join(lines)
    with open(path, 'w') as f:
        f.write(text)
    print("\n" + text)


def run_real():
    if h5py is None:
        raise RuntimeError("h5py is required for real-data runs (pip install h5py).")
    all_rows = []
    for dataset_name, loader in DATASET_LOADERS.items():
        print(f"\nLoading paired pixel<->latent samples for {dataset_name} ...")
        pairs = loader()
        print(f"  {len(pairs)} paired samples loaded.")
        if not pairs:
            continue
        wavelet = WAVELET_DICT[dataset_name]
        level = LEVEL_DICT[dataset_name]
        rows = analyze_dataset(dataset_name, pairs, wavelet, level)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No data processed -- check paths at the top of this file.")
        return
    df.to_csv(OUT_CSV, index=False)
    print(f"\nFull table written to {OUT_CSV}\n")
    print_rebuttal_table(df)
    write_rebuttal_summary(df, OUT_SUMMARY_TXT)
    print(f"\nCompact summary written to {OUT_SUMMARY_TXT}")


# ============================================================
# SELF-TEST -- synthetic H5 files with the exact schemas used above,
# built with a "storm" (smooth, low-frequency, high-autocorr) + "turbulence"
# (iid, decorrelated) synthetic field, so subband behavior has a known,
# checkable direction (LL should stay strongly autocorrelated; LH/HL/HH
# should collapse toward ~0). Run this in an environment with real h5py +
# pywt + torch installed to validate the full pipeline before trusting it
# on real data.
# ============================================================

def _make_synthetic_field(H, W, rng):
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = rng.uniform(0.3 * H, 0.7 * H), rng.uniform(0.3 * W, 0.7 * W)
    sigma = rng.uniform(0.1 * H, 0.2 * H)
    storm = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    noise = rng.normal(0, 0.15, size=(H, W))
    return (storm + noise).astype(np.float32)


def _build_selftest_h5(tmpdir, n_samples=20, T=15, H=128, W=128, C=4, f=4, seed=0):
    rng = np.random.default_rng(seed)
    pix_path = os.path.join(tmpdir, 'selftest_pixel.h5')
    lat_path = os.path.join(tmpdir, 'selftest_latent.h5')
    with h5py.File(pix_path, 'w') as fp, h5py.File(lat_path, 'w') as fl:
        gp = fp.create_group('train')
        gl = fl.create_group('train')
        for i in range(n_samples):
            key = f"sample_{i:04d}"
            seq = np.stack([_make_synthetic_field(H, W, rng) for _ in range(T)], axis=0)
            gp.create_dataset(key, data=seq)
            h, w = H // f, W // f
            lat_seq = np.zeros((T, C, h, w), dtype=np.float32)
            for t in range(T):
                base = seq[t].reshape(h, f, w, f).mean(axis=(1, 3))
                for c in range(C):
                    leak = 0.1 * np.roll(base, 1 if c % 2 else -1, axis=0)
                    lat_seq[t, c] = base * rng.uniform(0.8, 1.2) + leak + rng.normal(0, 0.03, size=base.shape)
            gl.create_dataset(key, data=lat_seq)
    return pix_path, lat_path


def run_selftest():
    import tempfile
    print("Running self-test with synthetic data (no real paths touched) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        pix_path, lat_path = _build_selftest_h5(tmpdir)

        def identity_preprocess(frame_hw):
            return frame_hw.astype(np.float32)  # already 128x128 in the synthetic set

        pairs = load_pairs_grouped_h5(pix_path, lat_path, is_cikm=True,
                                       preprocess_fn=identity_preprocess,
                                       max_samples=200, stride=1)
        print(f"  self-test: {len(pairs)} synthetic paired samples loaded.")
        rows = analyze_dataset('SELFTEST', pairs, wavelet='db6', level=1)
        df = pd.DataFrame(rows)
        print_rebuttal_table(df)

        ll_row = df[df['subband'] == 'LL'].iloc[0]
        hh_row = df[df['subband'] == 'HH'].iloc[0]
        assert ll_row['AC_pixel_mean'] > 0.5, "expected LL pixel autocorr to be strongly positive"
        assert ll_row['AC_latent_pool_mean'] > 0.5, "expected LL latent autocorr to be strongly positive"
        assert abs(hh_row['AC_pixel_mean']) < ll_row['AC_pixel_mean'], "expected HH << LL in pixel space"
        assert ll_row['LFE_pixel'] > ll_row['HFE_pixel'], "expected LL to concentrate low-frequency energy"
        print("\nSelf-test PASSED: expected LL-vs-HF ordering reproduced in both "
              "pixel and latent space on synthetic data.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--selftest', action='store_true',
                         help='Run on synthetic data only, to validate the pipeline.')
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        run_real()