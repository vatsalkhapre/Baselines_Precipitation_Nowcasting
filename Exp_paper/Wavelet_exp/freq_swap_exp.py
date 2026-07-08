#!/usr/bin/env python3
"""
frequency_swap_experiment.py
============================================================================
Tests the DAWN-Cast hypothesis:
        low  wavelet frequency (LL)      <-> convective bulk (slow, large-scale)
        high wavelet frequency (LH/HL/HH)<-> turbulence (fast, fine-scale)

Method (per event; N-frame sequence, frames indexed 1..N; last_k defaults to 5):
  * Grab LL(frame 1)   and  HIGH(frame 1).
  * For the last k frames t in {N-k+1 .. N}:
        SET A  "frozen bulk"       = IDWT( LL(1) , HIGH(t) )
        SET B  "frozen turbulence" = IDWT( LL(t) , HIGH(1) )
  * Plot per event:  frame 1 (source) | ground-truth last-k | SET A | SET B

  HIGH = coeffs[1:] = ALL detail tuples across ALL levels (paper Eq. 1 split:
  S_J = {LL_J} u U_{j=1..J} {LH_j,HL_j,HH_j}). Only the single coarsest
  approximation LL_J crosses between frames; the entire detail list is swapped
  as a whole, level-for-level.

Interpretation:
  Hypothesis holds if:
    - SET A (LL pinned to t=1)  -> storm looks FROZEN in place: large blob stays
      at frame-1 position/shape, only fine texture flickers  => bulk lives in LL.
    - SET B (HIGH pinned to t=1)-> storm EVOLVES like ground truth: blob moves/
      grows, fine texture pinned to frame-1 speckle  => large-scale motion in LL.
  The opposite behaviour refutes it. (A faint high-freq edge halo at the moving
  bulk's new location in SET A is expected -- detail bands encode edges too.)

Usage:
  SEVIR :  python frequency_swap_experiment.py --dataset sevir \
               --dataset_dir /home/vatsal/.../sevir --n_events 15
  CIKM  :  python frequency_swap_experiment.py --dataset cikm \
               --dataset_dir /home/vatsal/.../cikm.h5 --n_events 15
  Demo  :  python frequency_swap_experiment.py --demo --dataset {sevir|cikm}
           (no data / pywt / torch needed; Haar fallback + right colormap)
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
# Wavelet transform: pywt if available (db6, any level), else numpy Haar (demo).
# --------------------------------------------------------------------------
try:
    import pywt
    _HAVE_PYWT = True
except Exception:
    _HAVE_PYWT = False


def _haar_dwt2(x):
    a = (x[0::2, :] + x[1::2, :]) / np.sqrt(2.0)
    d = (x[0::2, :] - x[1::2, :]) / np.sqrt(2.0)
    LL = (a[:, 0::2] + a[:, 1::2]) / np.sqrt(2.0)
    LH = (a[:, 0::2] - a[:, 1::2]) / np.sqrt(2.0)
    HL = (d[:, 0::2] + d[:, 1::2]) / np.sqrt(2.0)
    HH = (d[:, 0::2] - d[:, 1::2]) / np.sqrt(2.0)
    return LL, (LH, HL, HH)


def _haar_idwt2(LL, highs):
    LH, HL, HH = highs
    a = np.zeros((LL.shape[0], LL.shape[1] * 2)); d = np.zeros_like(a)
    a[:, 0::2] = (LL + LH) / np.sqrt(2.0); a[:, 1::2] = (LL - LH) / np.sqrt(2.0)
    d[:, 0::2] = (HL + HH) / np.sqrt(2.0); d[:, 1::2] = (HL - HH) / np.sqrt(2.0)
    x = np.zeros((a.shape[0] * 2, a.shape[1]))
    x[0::2, :] = (a + d) / np.sqrt(2.0); x[1::2, :] = (a - d) / np.sqrt(2.0)
    return x


def decompose(frame, wavelet="db6", level=1, mode="periodization"):
    if _HAVE_PYWT:
        coeffs = pywt.wavedec2(frame, wavelet=wavelet, level=level, mode=mode)
        return coeffs[0], coeffs[1:]          # cA_J , [ALL detail levels]
    LL, highs = _haar_dwt2(frame)
    return LL, [highs]


def reconstruct(LL, details, wavelet="db6", mode="periodization"):
    if _HAVE_PYWT:
        return pywt.waverec2([LL] + list(details), wavelet=wavelet, mode=mode)
    return _haar_idwt2(LL, details[0])


def swap(frame_ll_src, frame_high_src, wavelet="db6", level=1):
    """LL from frame_ll_src, ALL detail bands (coeffs[1:]) from frame_high_src."""
    LL, _ = decompose(frame_ll_src, wavelet, level)
    _, details = decompose(frame_high_src, wavelet, level)
    return reconstruct(LL, details, wavelet)


# --------------------------------------------------------------------------
# Colormaps (match the paper / dataset files)
# --------------------------------------------------------------------------
_SEVIR_COLORS = [
    [0, 0, 0], [0.302, 0.302, 0.302], [0.157, 0.745, 0.157],
    [0.098, 0.588, 0.098], [0.039, 0.412, 0.039], [0.039, 0.294, 0.039],
    [0.961, 0.961, 0.0], [0.929, 0.675, 0.0], [0.941, 0.431, 0.0],
    [0.627, 0.0, 0.0], [0.906, 0.0, 1.0],
]
_SEVIR_BOUNDS = [0.0, 16, 31, 59, 74, 100, 133, 160, 181, 219, 255.0]

_CIKM_COLORS = np.array([
    [0, 0, 0, 0], [0, 236, 236, 255], [1, 160, 246, 255], [1, 0, 246, 255],
    [0, 239, 0, 255], [0, 200, 0, 255], [0, 144, 0, 255], [255, 255, 0, 255],
    [231, 192, 0, 255], [255, 144, 2, 255], [255, 0, 0, 255], [166, 0, 0, 255],
    [101, 0, 0, 255], [255, 0, 255, 255], [153, 85, 201, 255],
    [255, 255, 255, 255],
]) / 255
_CIKM_BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80.0]


def cmap_spec(kind):
    """Return (imshow_kwargs, top_value) for a given colormap kind."""
    if kind == "sevir":
        cmap = colors.ListedColormap(_SEVIR_COLORS)
        norm = colors.BoundaryNorm(_SEVIR_BOUNDS, cmap.N)
        return dict(cmap=cmap, norm=norm), _SEVIR_BOUNDS[-1]
    if kind == "cikm":
        cmap = colors.ListedColormap(_CIKM_COLORS)
        norm = colors.BoundaryNorm(_CIKM_BOUNDS, cmap.N)
        return dict(cmap=cmap, norm=norm), _CIKM_BOUNDS[-1]
    return dict(cmap="turbo", vmin=0, vmax=255.0), 255.0   # demo/turbo


# --------------------------------------------------------------------------
# Per-event figure
# --------------------------------------------------------------------------
def make_event_figure(seq, event_name, out_path, cmap_kind="sevir",
                      wavelet="db6", level=1, last_k=5):
    """
    seq : (T, H, W) already scaled into the colormap's value range.
    3 rows x (1+last_k): [GT, Frozen bulk LL(1)+HIGH(t), Frozen turb LL(t)+HIGH(1)]
    col 0 = source frame t=1 (shared anchor); cols = last_k frames.
    """
    T = seq.shape[0]
    f1 = seq[0]
    last_idx = list(range(T - last_k, T))
    gt   = [seq[i] for i in last_idx]
    setA = [swap(f1, seq[i], wavelet, level) for i in last_idx]  # LL(1)+HIGH(t)
    setB = [swap(seq[i], f1, wavelet, level) for i in last_idx]  # LL(t)+HIGH(1)

    imk, top = cmap_spec(cmap_kind)

    def prep(img):
        img = np.asarray(img)[: f1.shape[0], : f1.shape[1]]   # trim db6 edge padding
        return np.clip(img, 0, top)

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
# Synthetic demo (bulk drift+grow + decorrelated fine speckle)
# --------------------------------------------------------------------------
def synthetic_sequence(T=25, H=128, W=128, seed=0):
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:H, 0:W]
    seq = np.zeros((T, H, W), np.float32)
    for t in range(T):
        cy, cx, sig = 30 + 2.4 * t, 30 + 2.0 * t, 12 + 0.5 * t
        bulk = 210 * np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * sig ** 2)))
        fine = rng.normal(0, 1, (H, W)).astype(np.float32); fine -= fine.mean()
        seq[t] = np.clip(bulk + 60 * fine * (bulk > 20), 0, 255)
    return seq


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def run_sevir(args):
    import sys; sys.path.insert(0, args.code_dir)
    from dataset_sevir import SEVIRTorchDataset
    storm_filter = (lambda c: (c.pct_missing == 0) &
                    c.id.astype(str).str.startswith("S"))
    ds = SEVIRTorchDataset(
        dataset_dir=args.dataset_dir, seq_len=25, raw_seq_len=49,
        img_size=args.img_size, sample_mode="sequent", stride=args.stride,
        layout="NTHW", catalog_filter=storm_filter, shuffle=False,
        preprocess=True, rescale_method="01", split="test")
    print(f"[info] SEVIR storm sequences: {len(ds)}")
    _, top = cmap_spec("sevir")
    os.makedirs(args.out_dir, exist_ok=True)
    n = min(args.n_events, len(ds))
    for k in range(n):
        seq = np.squeeze(np.asarray(ds[k]))          # (25,H,W)
        assert seq.ndim == 3 and seq.shape[0] >= 25, f"bad shape {seq.shape}"
        seq = seq[:25] * top                          # [0,1] -> [0,255]
        out = os.path.join(args.out_dir, f"sevir_storm_{k:02d}.png")
        make_event_figure(seq, f"SEVIR storm #{k}", out, cmap_kind="sevir",
                          wavelet=args.wavelet, level=args.level, last_k=5)
        print(f"[saved] {out}")
    print(f"[done] {n} figures in {args.out_dir}")


def score_sequence(seq, active_thresh=50.0):
    """
    Catalog-free 'storm-likeness' score for a raw (T,H,W) CIKM sample (0-255 scale).
    CIKM has no CATALOG.csv / event metadata like SEVIR, so storm-like events must
    be found from the data itself. active_thresh is on the FIXED raw 0-255 scale,
    not a per-sample percentile -- a percentile-relative threshold lets wide, weak
    drizzle fields masquerade as high-coverage storms.
    """
    seq = seq.astype(np.float32)
    frac_active = (seq > active_thresh).mean()
    max_intensity = seq.max()
    p95_intensity = np.percentile(seq, 95)

    def centroid(frame, thr=active_thresh):
        m = np.where(frame > thr, frame, 0.0)
        mass = m.sum()
        if mass <= 1e-6:
            return None
        ys, xs = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
        return (ys * m).sum() / mass, (xs * m).sum() / mass

    c0, c1 = centroid(seq[0]), centroid(seq[-1])
    motion = 0.0
    if c0 is not None and c1 is not None:
        motion = float(np.hypot(c0[0] - c1[0], c0[1] - c1[1]))

    return dict(frac_active=float(frac_active), max_intensity=float(max_intensity),
                p95=float(p95_intensity), motion=motion)


def combined_score(m, scale_max=255.0):
    # Intensity gates out drizzle/noise; motion is a bonus so selected events
    # actually show bulk translation, which is what makes the frozen-bulk vs
    # frozen-turbulence swap visually legible.
    intensity_term = (m["max_intensity"] / scale_max) * 2.0 + (m["p95"] / scale_max) * 1.5
    coverage_term = m["frac_active"] * 1.0
    motion_term = np.tanh(m["motion"] / 15.0) * 1.0
    return intensity_term + coverage_term + motion_term


def select_storm_like_cikm(data_path, split, n_select, scan_limit=None,
                           active_thresh=50.0, seed=0):
    """
    Scans CIKM h5 samples directly (no catalog needed) and ranks them by a
    catalog-free storm-likeness score. Returns a sorted list of 0-indexed
    sample indices (best-first).
    """
    import h5py
    with h5py.File(data_path, "r") as f:
        total = int(f[split + "_len"][()])
        n_scan = total if scan_limit is None else min(scan_limit, total)
        rng = np.random.default_rng(seed)
        scan_idx = (np.arange(n_scan) if scan_limit is None
                    else rng.choice(total, size=n_scan, replace=False))
        scored = []
        for idx in scan_idx:
            key = f"sample_{int(idx) + 1}"
            seq = f[split][key][()]                    # (15,101,101) raw 0-255
            m = score_sequence(seq, active_thresh=active_thresh)
            scored.append((combined_score(m), int(idx)))
    scored.sort(key=lambda x: -x[0])
    return [idx for _, idx in scored[:n_select]]


def run_cikm(args):
    import sys; sys.path.insert(0, args.code_dir)
    from datasets.dataset_cikm import CIKM
    ds = CIKM(data_path=args.dataset_dir, type="test", img_size=args.img_size)
    print(f"[info] CIKM test samples: {len(ds)}  (no catalog.csv -- CIKM has no "
          f"event metadata, so selection is data-driven)")
    _, top = cmap_spec("cikm")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.select == "storm_like":
        idxs = select_storm_like_cikm(args.dataset_dir, split="test",
                                      n_select=args.n_events,
                                      scan_limit=args.scan_limit,
                                      active_thresh=args.active_thresh)
        print(f"[info] storm-like indices (data-driven, best-first): {idxs}")
    else:
        n = min(args.n_events, len(ds))
        idxs = np.linspace(0, len(ds) - 1, n).astype(int).tolist()

    for k, idx in enumerate(idxs):
        seq = np.squeeze(np.asarray(ds[int(idx)]))    # [15,1,W,H] -> (15,W,H)
        assert seq.ndim == 3 and seq.shape[0] >= 15, f"bad shape {seq.shape}"
        seq = seq[:15] * top                           # [0,1] -> [0,80] dBZ (colorbar scale)
        out = os.path.join(args.out_dir, f"cikm_{k:02d}_s{idx}.png")
        make_event_figure(seq, f"CIKM sample #{idx}", out, cmap_kind="cikm",
                          wavelet=args.wavelet, level=args.level, last_k=5)
        print(f"[saved] {out}")
    print(f"[done] {len(idxs)} figures in {args.out_dir}")


def run_demo(args):
    os.makedirs(args.out_dir, exist_ok=True)
    T = 15 if args.dataset == "cikm" else 25
    kind = args.dataset if args.dataset in ("sevir", "cikm") else "turbo"
    _, top = cmap_spec(kind)
    for k in range(args.n_events):
        seq = synthetic_sequence(T=T, seed=k)          # 0..255
        if kind == "cikm":
            seq = seq / 255.0 * top                     # rescale into dBZ bounds
        out = os.path.join(args.out_dir, f"demo_{args.dataset}_{k:02d}.png")
        make_event_figure(seq, f"SYNTHETIC {args.dataset} #{k}", out,
                          cmap_kind=kind, wavelet=args.wavelet,
                          level=args.level, last_k=5)
        print(f"[saved] {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sevir", "cikm"], default="sevir")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--dataset_dir", type=str,
                   default="/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/sevir",
                   help="SEVIR: root dir with CATALOG.csv + data/.  CIKM: path to cikm.h5")
    p.add_argument("--code_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, default="./freq_swap_figs")
    p.add_argument("--n_events", type=int, default=15)
    p.add_argument("--wavelet", type=str, default="db6")
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--stride", type=int, default=49)     # SEVIR: ~1 window/event
    p.add_argument("--img_size", type=int, default=384)  # SEVIR 384; CIKM use 128
    p.add_argument("--select", choices=["storm_like", "linspace"], default="storm_like",
                   help="CIKM only: how to pick events (no catalog.csv exists for CIKM, "
                        "so 'storm_like' scores samples directly from the data).")
    p.add_argument("--scan_limit", type=int, default=None,
                   help="CIKM only: cap how many test samples to scan when scoring "
                        "(default None = scan all; use e.g. 1000 for a faster, "
                        "random-subset scan on very large test splits).")
    p.add_argument("--active_thresh", type=float, default=50.0,
                   help="CIKM only: fixed raw-intensity threshold (0-255 scale) "
                        "used to decide 'active' precipitation pixels for scoring.")
    args = p.parse_args()

    if args.demo:
        run_demo(args)
    elif args.dataset == "sevir":
        run_sevir(args)
    else:
        run_cikm(args)


if __name__ == "__main__":
    main()