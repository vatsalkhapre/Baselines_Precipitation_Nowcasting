#!/usr/bin/env python3
"""
frequency_swap_experiment.py
============================================================================
Tests the DAWN-Cast hypothesis:
        low  wavelet frequency (LL)   <-> convective bulk (slow, large-scale)
        high wavelet frequency (LHHLHH) <-> turbulence (fast, fine-scale)

Method (per event, 25-frame sequence, frames indexed 1..25):
  * Grab LL(frame 1)   and  HIGH(frame 1).
  * For the last 5 frames t in {21,22,23,24,25}:
        SET A  "frozen bulk"       = IDWT( LL(1)   , HIGH(t) )
        SET B  "frozen turbulence" = IDWT( LL(t)   , HIGH(1) )
  * Plot, per event:  frame 1 (source) | ground-truth last-5 | SET A | SET B

Interpretation:
  If the hypothesis holds:
    - SET A (LL pinned to t=1) should look like the storm *frozen in place*:
      the large-scale blob stays at its frame-1 position/shape while only fine
      texture flickers  =>  bulk lives in LL.
    - SET B (HIGH pinned to t=1) should *evolve like ground truth*: the blob
      moves/grows following the true large-scale motion while the fine texture
      is pinned to frame-1 speckle  =>  large-scale motion lives in LL, not HIGH.
  The opposite behaviour would refute the hypothesis.

Runs two ways:
  1. Real data :  python frequency_swap_experiment.py \
                      --dataset_dir /home/vatsal/.../sevir \
                      --n_events 15 --wavelet db6 --level 1
  2. Synthetic :  python frequency_swap_experiment.py --demo
     (no SEVIR, no pywt, no torch needed; uses a built-in Haar fallback)
============================================================================
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors

# --------------------------------------------------------------------------
# Wavelet transform:  pywt if available (db6, any level),
# else a pure-numpy single-level Haar fallback (used by --demo).
# --------------------------------------------------------------------------
try:
    import pywt
    _HAVE_PYWT = True
except Exception:
    _HAVE_PYWT = False


def _haar_dwt2(x):
    """Single-level Haar DWT (periodization). x: (H,W), H,W even."""
    a = (x[0::2, :] + x[1::2, :]) / np.sqrt(2.0)      # rows low
    d = (x[0::2, :] - x[1::2, :]) / np.sqrt(2.0)      # rows high
    LL = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2.0)
    LH = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2.0)
    HL = (d[:, 0::2] + d[:, 1::2]) / np.sqrt(2.0)
    HH = (d[:, 0::2] - d[:, 1::2]) / np.sqrt(2.0)
    return LL, (LH, HL, HH)


def _haar_idwt2(LL, highs):
    LH, HL, HH = highs
    a = np.zeros((LL.shape[0], LL.shape[1] * 2))
    d = np.zeros_like(a)
    a[:, 0::2] = (LL + LH) / np.sqrt(2.0)
    a[:, 1::2] = (LL - LH) / np.sqrt(2.0)
    d[:, 0::2] = (HL + HH) / np.sqrt(2.0)
    d[:, 1::2] = (HL - HH) / np.sqrt(2.0)
    x = np.zeros((a.shape[0] * 2, a.shape[1]))
    x[0::2, :] = (a + d) / np.sqrt(2.0)
    x[1::2, :] = (a - d) / np.sqrt(2.0)
    return x


def decompose(frame, wavelet="db6", level=1, mode="periodization"):
    """Return (approx_LL, detail_coeffs) for a 2-D frame."""
    if _HAVE_PYWT:
        coeffs = pywt.wavedec2(frame, wavelet=wavelet, level=level, mode=mode)
        return coeffs[0], coeffs[1:]          # cA_n , [ (cH,cV,cD), ... ]
    LL, highs = _haar_dwt2(frame)
    return LL, [highs]


def reconstruct(LL, details, wavelet="db6", mode="periodization"):
    """Inverse transform from an approximation band + detail bands."""
    if _HAVE_PYWT:
        return pywt.waverec2([LL] + list(details), wavelet=wavelet, mode=mode)
    return _haar_idwt2(LL, details[0])


def swap_frozen_bulk(frame_ll_src, frame_high_src, wavelet="db6", level=1):
    """LL from frame_ll_src, ALL detail bands from frame_high_src."""
    LL, _ = decompose(frame_ll_src, wavelet, level)
    _, details = decompose(frame_high_src, wavelet, level)
    return reconstruct(LL, details, wavelet)


def swap_frozen_turb(frame_ll_src, frame_high_src, wavelet="db6", level=1):
    """LL from frame_ll_src, ALL detail bands from frame_high_src (same helper,
    named for clarity at call sites)."""
    return swap_frozen_bulk(frame_ll_src, frame_high_src, wavelet, level)


# --------------------------------------------------------------------------
# SEVIR colour map (matches the paper figures)
# --------------------------------------------------------------------------
_SEVIR_COLORS = [
    [0, 0, 0], [0.302, 0.302, 0.302], [0.157, 0.745, 0.157],
    [0.098, 0.588, 0.098], [0.039, 0.412, 0.039], [0.039, 0.294, 0.039],
    [0.961, 0.961, 0.0], [0.929, 0.675, 0.0], [0.941, 0.431, 0.0],
    [0.627, 0.0, 0.0], [0.906, 0.0, 1.0],
]
_SEVIR_BOUNDS = [0.0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255.0]


def sevir_cmap_norm():
    cmap = colors.ListedColormap(_SEVIR_COLORS)
    norm = colors.BoundaryNorm(_SEVIR_BOUNDS, cmap.N)
    return cmap, norm


# --------------------------------------------------------------------------
# Per-event figure
# --------------------------------------------------------------------------
def make_event_figure(seq, event_name, out_path, wavelet="db6", level=1,
                      last_k=5, use_sevir_cmap=True, vmax=255.0):
    """
    seq : (T, H, W) float array, T >= 25.
    Produces a 3-row x (1+last_k) grid:
        rows  = [Ground truth, Frozen bulk (LL@1), Frozen turbulence (HIGH@1)]
        col 0 = source frame t=1  (shared anchor)
        cols  = last_k predicted frames (t = T-last_k+1 .. T)
    """
    T = seq.shape[0]
    f1 = seq[0]
    last_idx = list(range(T - last_k, T))            # e.g. 20..24 (frames 21..25)
    gt = [seq[i] for i in last_idx]

    setA = [swap_frozen_bulk(f1, seq[i], wavelet, level) for i in last_idx]  # LL(1)+HIGH(t)
    setB = [swap_frozen_turb(seq[i], f1, wavelet, level) for i in last_idx]  # LL(t)+HIGH(1)

    def prep(img):
        img = np.asarray(img)
        img = img[: f1.shape[0], : f1.shape[1]]      # guard db6 edge padding
        return np.clip(img, 0, vmax)

    if use_sevir_cmap:
        cmap, norm = sevir_cmap_norm()
        imk = dict(cmap=cmap, norm=norm)
    else:
        imk = dict(cmap="turbo", vmin=0, vmax=vmax)

    ncol = 1 + last_k
    fig, ax = plt.subplots(3, ncol, figsize=(2.05 * ncol, 6.4))
    rows = [
        ("Ground truth", [f1] + gt),
        ("Frozen bulk\nLL(1)+HIGH(t)", [f1] + setA),
        ("Frozen turb.\nLL(t)+HIGH(1)", [f1] + setB),
    ]
    col_titles = ["t=1 (source)"] + [f"t={i+1}" for i in last_idx]

    for r, (rlabel, imgs) in enumerate(rows):
        for c in range(ncol):
            a = ax[r, c]
            a.imshow(prep(imgs[c]), **imk)
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(col_titles[c], fontsize=9)
            if c == 0:
                a.set_ylabel(rlabel, fontsize=9)
            if c == 0 and r > 0:
                for s in a.spines.values():
                    s.set_edgecolor("0.6"); s.set_linestyle("--")

    fig.suptitle(f"Frequency-swap test  |  {event_name}  "
                 f"(wavelet={wavelet if _HAVE_PYWT else 'haar'}, level={level})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Synthetic demo: bulk (smooth translating/growing blob) + turbulence (speckle)
# --------------------------------------------------------------------------
def synthetic_sequence(T=25, H=128, W=128, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W]
    seq = np.zeros((T, H, W), np.float32)
    # turbulence: a fine, fast-decorrelating speckle field per frame
    for t in range(T):
        # bulk: a Gaussian blob that drifts + grows (large-scale, slow)
        cy = 30 + 2.4 * t
        cx = 30 + 2.0 * t
        sig = 12 + 0.5 * t
        bulk = 210 * np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig ** 2)))
        # fine turbulence, re-drawn each frame (decorrelated in time), masked to storm
        fine = rng.normal(0, 1, (H, W)).astype(np.float32)
        fine = fine - fine.mean()
        speckle = 60 * fine * (bulk > 20)
        seq[t] = np.clip(bulk + speckle, 0, 255)
    return seq


# --------------------------------------------------------------------------
def run_real(args):
    import torch  # noqa
    import sys
    sys.path.insert(0, args.code_dir)
    from datasets.dataset_sevir import SEVIRTorchDataset

    storm_filter = (lambda c: (c.pct_missing == 0) &
                    c.id.astype(str).str.startswith("S"))

    ds = SEVIRTorchDataset(
        dataset_dir=args.dataset_dir,
        seq_len=25, raw_seq_len=49, img_size=args.img_size,
        sample_mode="sequent", stride=args.stride, layout="NTHW",
        catalog_filter=storm_filter, shuffle=False, preprocess=True,
        rescale_method="01", split="test",
    )
    print(f"[info] storm sequences available: {len(ds)}")
    os.makedirs(args.out_dir, exist_ok=True)

    n = min(args.n_events, len(ds))
    for k in range(n):
        item = np.asarray(ds[k])                 # (1,25,1,H,W)
        seq = np.squeeze(item)                   # (25,H,W)
        assert seq.ndim == 3 and seq.shape[0] >= 25, f"bad shape {seq.shape}"
        seq = seq[:25] * 255.0                    # rescale_method='01' -> 0..255 for cmap
        out = os.path.join(args.out_dir, f"storm_{k:02d}.png")
        make_event_figure(seq, f"SEVIR storm #{k}", out,
                          wavelet=args.wavelet, level=args.level,
                          use_sevir_cmap=True, vmax=255.0)
        print(f"[saved] {out}")
    print(f"[done] {n} figures in {args.out_dir}")


def run_demo(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for k in range(args.n_events):
        seq = synthetic_sequence(T=25, seed=k)
        out = os.path.join(args.out_dir, f"synthetic_{k:02d}.png")
        make_event_figure(seq, f"SYNTHETIC (bulk+turbulence) #{k}", out,
                          wavelet=args.wavelet, level=args.level,
                          use_sevir_cmap=False, vmax=255.0)
        print(f"[saved] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true",
                   help="Run on synthetic data (no SEVIR / pywt / torch needed).")
    p.add_argument("--dataset_dir", type=str,
                   default="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/sevir")
    p.add_argument("--code_dir", type=str, default=".",
                   help="Directory containing dataset_sevir.py")
    p.add_argument("--out_dir", type=str, default="./freq_swap_figs")
    p.add_argument("--n_events", type=int, default=15)
    p.add_argument("--wavelet", type=str, default="db6")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--stride", type=int, default=49)   # ~1 window/event -> diverse storms
    p.add_argument("--img_size", type=int, default=384)
    args = p.parse_args()

    if args.demo:
        run_demo(args)
    else:
        run_real(args)


if __name__ == "__main__":
    main()