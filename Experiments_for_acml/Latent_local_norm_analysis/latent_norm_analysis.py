"""
latent_norm_analysis.py
========================
Experiments to justify *latent-space standardization* for precipitation nowcasting.

Produces, for each dataset:
  (1) 1D distribution + CDF of PIXEL-space values            (scaled, e.g. [0,1])
  (2) 1D distribution + CDF of UNNORMALIZED latent values    (~[-15,15])
  (3) 1D distribution + CDF of NORMALIZED latent values      (latent / global_std)
  (4) 2D per-pixel(/per-channel) mean & std maps for all three spaces,
      plus the *aggregate* (global) mean / std / skew / kurtosis printed + saved to JSON.

It is written to plug into your existing `get_dataset(...)` factory, but it does NOT
import it directly (to avoid the relative-import / heavy-deps problem). Instead it
re-implements the *minimal* loading path for each dataset, matching your dataset
classes exactly. Set DATASET below and point PATHS at your .h5 files / dirs.

Design choices that matter for the paper (read these):
  * The normalized latent is divided by a SINGLE GLOBAL std estimated over a sample of
    the TRAIN split (this is the data-driven analogue of Stable Diffusion's fixed
    `scale_factor = 0.18215`; see Rombach et al., LDM, CVPR 2022). We ALSO report the
    distribution of per-sample stds so you can show the per-sample std you use at test
    time is tightly concentrated around the global value -> the two are interchangeable.
  * Latent stats are reported BOTH globally and PER-CHANNEL, because the real story is
    that channels have *unequal* variance and standardization equalizes them.
  * Everything is computed in a streaming / running-moments fashion so it scales to the
    full dataset without loading it all into RAM.
"""

import os
import os.path as osp
import json
import argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- #
#  Running moment accumulators (Welford / streaming) -- scale to full dataset
# ----------------------------------------------------------------------------- #
class ScalarMoments:
    """Streaming mean/var/min/max/skew/kurtosis over a flat stream of values,
    plus a subsampled reservoir of raw values for histogram/CDF plotting."""
    def __init__(self, reservoir_size=2_000_000, seed=0):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0
        self.vmin = np.inf
        self.vmax = -np.inf
        self.reservoir = []
        self.reservoir_size = reservoir_size
        self.rng = np.random.default_rng(seed)

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        if x.size == 0:
            return
        self.vmin = min(self.vmin, float(x.min()))
        self.vmax = max(self.vmax, float(x.max()))
        # batch update of central moments (Pebay 2008 parallel formulas, done per-batch)
        for chunk in np.array_split(x, max(1, x.size // 100000)):
            self._update_chunk(chunk)
        # reservoir sampling for plotting
        self._reservoir_add(x)

    def _update_chunk(self, x):
        nB = x.size
        if nB == 0:
            return
        meanB = x.mean()
        dB = x - meanB
        M2B = (dB**2).sum()
        M3B = (dB**3).sum()
        M4B = (dB**4).sum()
        nA = self.n
        if nA == 0:
            self.n, self.mean, self.M2, self.M3, self.M4 = nB, meanB, M2B, M3B, M4B
            return
        delta = meanB - self.mean
        nAB = nA + nB
        # combine
        new_mean = self.mean + delta * nB / nAB
        M2 = self.M2 + M2B + delta**2 * nA * nB / nAB
        M3 = (self.M3 + M3B
              + delta**3 * nA * nB * (nA - nB) / nAB**2
              + 3 * delta * (nA * M2B - nB * self.M2) / nAB)
        M4 = (self.M4 + M4B
              + delta**4 * nA * nB * (nA**2 - nA*nB + nB**2) / nAB**3
              + 6 * delta**2 * (nA**2 * M2B + nB**2 * self.M2) / nAB**2
              + 4 * delta * (nA * M3B - nB * self.M3) / nAB)
        self.n, self.mean, self.M2, self.M3, self.M4 = nAB, new_mean, M2, M3, M4

    def _reservoir_add(self, x):
        # simple: keep a random subsample, cap at reservoir_size
        if len(self.reservoir) * 0 == 0:  # always
            take = x
            if take.size > 200000:
                idx = self.rng.choice(take.size, 200000, replace=False)
                take = take[idx]
            self.reservoir.append(take.astype(np.float32))
            total = sum(len(r) for r in self.reservoir)
            if total > self.reservoir_size:
                # collapse + subsample down
                allv = np.concatenate(self.reservoir)
                idx = self.rng.choice(allv.size, self.reservoir_size, replace=False)
                self.reservoir = [allv[idx]]

    @property
    def var(self):
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std(self):
        return float(np.sqrt(self.var))

    @property
    def skew(self):
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return float(np.sqrt(self.n) * self.M3 / (self.M2**1.5))

    @property
    def kurtosis(self):  # excess kurtosis
        if self.n < 2 or self.M2 == 0:
            return 0.0
        return float(self.n * self.M4 / (self.M2**2) - 3.0)

    def values(self):
        return np.concatenate(self.reservoir) if self.reservoir else np.array([])

    def summary(self):
        return dict(n=int(self.n), mean=float(self.mean), std=self.std,
                    var=float(self.var), min=float(self.vmin), max=float(self.vmax),
                    skew=self.skew, excess_kurtosis=self.kurtosis)


class SpatialMoments:
    """Running per-pixel (and per-channel) mean/std over [C,H,W] frames.
    Accumulates sum and sum-of-squares per element."""
    def __init__(self):
        self.count = 0
        self.sum = None
        self.sumsq = None

    def update(self, frame_chw):
        # frame_chw: np.ndarray [C,H,W]
        f = np.asarray(frame_chw, dtype=np.float64)
        if self.sum is None:
            self.sum = np.zeros_like(f)
            self.sumsq = np.zeros_like(f)
        self.sum += f
        self.sumsq += f**2
        self.count += 1

    def mean_map(self):
        return self.sum / max(1, self.count)

    def std_map(self):
        m = self.mean_map()
        var = self.sumsq / max(1, self.count) - m**2
        return np.sqrt(np.clip(var, 0, None))


# ----------------------------------------------------------------------------- #
#  Dataset iterators -- yield frames as torch tensors / numpy, matching your code
#  Each iterator yields a single sequence sample shaped [T, C, H, W] (pixel space
#  C=1) or [T, C, H, W] / [C,...] latent depending on how your latent .h5 stores it.
# ----------------------------------------------------------------------------- #
def iter_pixel_dataset(name, path, img_size, split, max_samples):
    """Pixel-space (scaled) frames, matching dataset_{cikm,shanghai,meteonet,sevir}.py."""
    import h5py
    from torchvision import transforms

    if name == "cikm":
        t = split if split != "valid" else "test"
        tf = transforms.CenterCrop((img_size, img_size))
        with h5py.File(path, "r") as f:
            n = 1000 if split == "valid" else int(f[split + "_len"][()])
            n = min(n, max_samples) if max_samples else n
            for i in range(n):
                imgs = f[t]["sample_" + str(i + 1)][()]
                seqs = torch.from_numpy(imgs).float()
                fr = tf(seqs) / 255.0
                yield fr.unsqueeze(1).numpy()           # [T,1,H,W]

    elif name in ("shanghai",):
        t = split if split != "val" else "test"
        tf = transforms.Resize((img_size, img_size))
        with h5py.File(path, "r") as f:
            n = int(f[t]["all_len"][()])
            n = min(n, max_samples) if max_samples else n
            for i in range(n):
                imgs = f[t][str(i)][()]
                fr = torch.from_numpy(imgs).float().squeeze() / 255.0
                fr = tf(fr)
                yield fr.unsqueeze(1).numpy()

    elif name in ("meteo", "meteonet"):
        t = split if split != "val" else "test"
        tf = transforms.Resize((img_size, img_size))
        PIXEL_SCALE = 90.0
        with h5py.File(path, "r") as f:
            n = int(f[f"{t}_len"][()])
            n = min(n, max_samples) if max_samples else n
            for i in range(n):
                imgs = f[t][str(i)][()]
                fr = torch.from_numpy(imgs).float().squeeze() / PIXEL_SCALE
                fr = tf(fr)
                yield fr.unsqueeze(1).numpy()
    else:
        raise ValueError(f"pixel iterator not implemented for {name}")


def iter_latent_dataset(name, path, split, max_samples):
    """Unnormalized latent frames, matching dataset_*_latent*.py.
    These return the raw stored latent (no /255, no transform)."""
    import h5py

    if name in ("cikm_latent_32", "cikm"):
        t = split if split != "valid" else "test"
        with h5py.File(path, "r") as f:
            n = 1000 if split == "valid" else int(f[split + "_len"][()])
            n = min(n, max_samples) if max_samples else n
            for i in range(n):
                arr = f[t]["sample_" + str(i + 1)][()]
                yield np.asarray(arr)                   # [T,C,h,w]

    elif name in ("shanghai_lr_latent_32", "shanghai"):
        t = split if split != "val" else "test"
        with h5py.File(path, "r") as f:
            n = int(f[t]["all_len"][()])
            n = min(n, max_samples) if max_samples else n
            for i in range(n):
                yield np.asarray(f[t][str(i)][()])

    elif name in ("meteo_lr_latent_32", "meteo", "meteonet"):
        t = split if split != "val" else "test"
        with h5py.File(path, "r") as f:
            n = int(f[f"{t}_len"][()])
            n = min(n, max_samples) if max_samples else n
            for i in range(n):
                yield np.asarray(f[t][str(i)][()])
    else:
        raise ValueError(f"latent iterator not implemented for {name}")


# ----------------------------------------------------------------------------- #
#  Helpers: normalize a [T,...] sequence into [N, C, H, W] frames
# ----------------------------------------------------------------------------- #
def to_frames_chw(seq):
    """Accept [T,C,H,W], [T,H,W], [C,H,W], or [T,1,H,W] -> list of [C,H,W]."""
    a = np.asarray(seq, dtype=np.float32)
    a = np.squeeze(a)
    if a.ndim == 2:                       # [H,W]
        return [a[None]]
    if a.ndim == 3:                       # [T,H,W] (pixel) OR [C,H,W] (single latent)
        # heuristic: small first dim & >1 -> treat as channels only if equals 4/8/16
        return [a[i][None] for i in range(a.shape[0])]  # treat first dim as frames
    if a.ndim == 4:                       # [T,C,H,W]
        T, C = a.shape[0], a.shape[1]
        return [a[t] for t in range(T)]
    raise ValueError(f"unexpected shape {a.shape}")


def latent_frames_chw(seq):
    """For latent: preserve channel dim. Accept [T,C,h,w] or [C,h,w] or [T,h,w]."""
    a = np.asarray(seq, dtype=np.float32)
    if a.ndim == 5:
        a = a.squeeze()
    if a.ndim == 4:                       # [T,C,h,w]
        return [a[t] for t in range(a.shape[0])]
    if a.ndim == 3:                       # ambiguous: [C,h,w] single frame OR [T,h,w]
        # If first dim in {4,8,16} assume channels (one frame); else frames of 1 channel
        if a.shape[0] in (3, 4, 8, 16):
            return [a]
        return [a[t][None] for t in range(a.shape[0])]
    raise ValueError(f"unexpected latent shape {a.shape}")


# ----------------------------------------------------------------------------- #
#  Pass 1: estimate global latent std (and per-channel std) over TRAIN sample
# ----------------------------------------------------------------------------- #
def estimate_latent_global_std(name, path, max_samples):
    sm = ScalarMoments(reservoir_size=1)  # we only need std here
    per_channel_sumsq = None
    per_channel_sum = None
    per_channel_count = 0
    per_sample_stds = []
    for seq in iter_latent_dataset(name, path, "train", max_samples):
        for fr in latent_frames_chw(seq):  # [C,h,w]
            sm.update(fr)
            C = fr.shape[0]
            flat = fr.reshape(C, -1).astype(np.float64)
            if per_channel_sum is None:
                per_channel_sum = np.zeros(C)
                per_channel_sumsq = np.zeros(C)
            per_channel_sum += flat.sum(1)
            per_channel_sumsq += (flat**2).sum(1)
            per_channel_count += flat.shape[1]
        per_sample_stds.append(float(np.asarray(seq, dtype=np.float64).std()))
    global_std = sm.std
    global_mean = sm.mean
    ch_mean = per_channel_sum / per_channel_count
    ch_std = np.sqrt(np.clip(per_channel_sumsq / per_channel_count - ch_mean**2, 0, None))
    return dict(global_std=global_std, global_mean=global_mean,
                per_channel_mean=ch_mean.tolist(), per_channel_std=ch_std.tolist(),
                per_sample_std_mean=float(np.mean(per_sample_stds)),
                per_sample_std_std=float(np.std(per_sample_stds)))


# ----------------------------------------------------------------------------- #
#  LOCAL-WINDOW std normalization  ==  the actual method:  enc(x_in)/enc(x_in).std()
#  For frame t, divisor = std over the 5-frame window:
#     centered  [t-2, t-1, t, t+1, t+2]  when t+2 <= last index
#     causal    [t-4, t-3, t-2, t-1, t]  when the centered window would overrun the end
#  std is taken over ALL pixels & channels in the window (one scalar per frame),
#  exactly matching frames_in.std().
# ----------------------------------------------------------------------------- #
def local_window_indices(t, T, half=2):
    """Return the list of frame indices used to compute frame t's divisor."""
    if t + half <= T - 1:                       # centered window fits
        lo = max(0, t - half)
        idx = list(range(lo, t + half + 1))
        # if we are near the START, pad forward so we keep 2*half+1 frames
        need = (2 * half + 1) - len(idx)
        if need > 0:
            idx = list(range(idx[0], min(T, idx[-1] + 1 + need)))
        return idx[:2 * half + 1]
    else:                                        # end of sequence -> causal prev-4 + self
        hi = t
        lo = max(0, hi - 2 * half)
        idx = list(range(lo, hi + 1))
        return idx


def local_window_std_normalize(seq_frames, half=2, eps=1e-6):
    """seq_frames: list of [C,h,w] frames for ONE sequence.
    Returns list of normalized frames and the list of per-frame divisors used."""
    T = len(seq_frames)
    stack = np.stack(seq_frames, axis=0).astype(np.float64)   # [T,C,h,w]
    out, divisors = [], []
    for t in range(T):
        idx = local_window_indices(t, T, half=half)
        window = stack[idx]                       # [w,C,h,w]
        s = float(window.std())                   # scalar over all pixels & channels
        s = s if s > eps else eps
        out.append((stack[t] / s).astype(np.float32))
        divisors.append(s)
    return out, divisors


# ----------------------------------------------------------------------------- #
#  Plotting
# ----------------------------------------------------------------------------- #
def plot_hist_cdf(moments, title, color, outpath, drop_zeros_note=""):
    vals = moments.values()
    if vals.size == 0:
        return
    s = moments.summary()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # histogram (density)
    ax1.hist(vals, bins=200, color=color, alpha=0.85, density=True)
    ax1.set_yscale("log")
    ax1.set_xlabel("value"); ax1.set_ylabel("density (log)")
    ax1.set_title(f"{title}\nmean={s['mean']:.3f}  std={s['std']:.3f}  "
                  f"skew={s['skew']:.2f}  exkurt={s['excess_kurtosis']:.2f}")
    ax1.axvline(s["mean"], color="k", ls="--", lw=1)
    # CDF
    sv = np.sort(vals)
    cdf = np.arange(1, sv.size + 1) / sv.size
    ax2.plot(sv, cdf, color=color, lw=1.5)
    ax2.set_xlabel("value"); ax2.set_ylabel("cumulative probability")
    ax2.set_title(f"CDF  {drop_zeros_note}")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_spatial(mean_map, std_map, title, outpath):
    # collapse channel dim for display (mean over channels) but report both
    mm = mean_map.mean(0) if mean_map.ndim == 3 else mean_map
    ss = std_map.mean(0) if std_map.ndim == 3 else std_map
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    im1 = a1.imshow(mm, cmap="viridis"); a1.set_title(f"{title}\nper-pixel MEAN")
    plt.colorbar(im1, ax=a1, fraction=0.046)
    im2 = a2.imshow(ss, cmap="magma"); a2.set_title("per-pixel STD")
    plt.colorbar(im2, ax=a2, fraction=0.046)
    for a in (a1, a2): a.axis("off")
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------- #
#  Main driver
# ----------------------------------------------------------------------------- #
def run(name_pixel, path_pixel, name_latent, path_latent, img_size,
        outdir, split, max_samples):
    os.makedirs(outdir, exist_ok=True)
    report = {}

    # ---- Pass 1: global latent std (the SD-style scale factor) ----
    print(f"[{name_latent}] estimating global latent std over train...")
    latent_norm_stats = estimate_latent_global_std(name_latent, path_latent, max_samples)
    g_std = latent_norm_stats["global_std"]
    report["latent_normalization"] = latent_norm_stats
    print(f"  global latent std = {g_std:.4f}  "
          f"(per-sample std = {latent_norm_stats['per_sample_std_mean']:.4f} "
          f"+/- {latent_norm_stats['per_sample_std_std']:.4f})")

    # ---- (1) PIXEL space ----
    print(f"[{name_pixel}] pixel-space pass...")
    px_scalar = ScalarMoments()
    px_spatial = SpatialMoments()
    for seq in iter_pixel_dataset(name_pixel, path_pixel, img_size, split, max_samples):
        for fr in to_frames_chw(seq):           # [1,H,W]
            px_scalar.update(fr)
            px_spatial.update(fr)
    report["pixel"] = px_scalar.summary()
    plot_hist_cdf(px_scalar, "Pixel space (scaled)", "#1f77b4",
                  osp.join(outdir, "1_pixel_dist.png"))
    plot_spatial(px_spatial.mean_map(), px_spatial.std_map(),
                 "Pixel space", osp.join(outdir, "1_pixel_spatial.png"))

    # ---- (2),(3),(3b) LATENT space ----
    #   lat_un  : unnormalized latent
    #   lat_lw  : LOCAL-WINDOW std normalized  ==  THE METHOD  (enc(x_in)/enc(x_in).std())
    #   lat_no  : global-std normalized        (secondary comparison)
    print(f"[{name_latent}] latent-space pass...")
    lat_un = ScalarMoments()
    lat_lw = ScalarMoments()        # local-window (primary)
    lat_no = ScalarMoments()        # global std  (secondary)
    sp_un = SpatialMoments()
    sp_lw = SpatialMoments()
    sp_no = SpatialMoments()
    all_divisors = []               # collect per-frame local-window stds for diagnostics
    for seq in iter_latent_dataset(name_latent, path_latent, split, max_samples):
        frames = latent_frames_chw(seq)              # list of [C,h,w]
        lw_frames, divisors = local_window_std_normalize(frames, half=2)
        all_divisors.extend(divisors)
        for fr, frlw in zip(frames, lw_frames):
            lat_un.update(fr)
            lat_lw.update(frlw)
            lat_no.update(fr / g_std)
            sp_un.update(fr)
            sp_lw.update(frlw)
            sp_no.update(fr / g_std)
    report["latent_unnormalized"] = lat_un.summary()
    report["latent_localwindow_norm"] = lat_lw.summary()
    report["latent_globalstd_norm"] = lat_no.summary()
    all_divisors = np.asarray(all_divisors)
    report["local_window_divisor"] = dict(
        mean=float(all_divisors.mean()), std=float(all_divisors.std()),
        min=float(all_divisors.min()), max=float(all_divisors.max()),
        global_std_for_reference=g_std,
        note="per-frame divisor = std over 5-frame local window (enc(x_in).std() analogue)")

    plot_hist_cdf(lat_un, "Latent (unnormalized)", "#d62728",
                  osp.join(outdir, "2_latent_unnorm_dist.png"))
    plot_hist_cdf(lat_lw, "Latent (local-window std norm)  [OUR METHOD]", "#9467bd",
                  osp.join(outdir, "3_latent_localwin_dist.png"))
    plot_hist_cdf(lat_no, "Latent (global std norm)", "#2ca02c",
                  osp.join(outdir, "3b_latent_globalstd_dist.png"))
    plot_spatial(sp_un.mean_map(), sp_un.std_map(),
                 "Latent (unnormalized)", osp.join(outdir, "2_latent_unnorm_spatial.png"))
    plot_spatial(sp_lw.mean_map(), sp_lw.std_map(),
                 "Latent (local-window norm)", osp.join(outdir, "3_latent_localwin_spatial.png"))
    plot_spatial(sp_no.mean_map(), sp_no.std_map(),
                 "Latent (global norm)", osp.join(outdir, "3b_latent_globalstd_spatial.png"))

    # ---- divisor histogram: shows the per-input std spans a wide range
    #      (calm vs intense precip regimes) -> "present climate context"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_divisors, bins=80, color="#9467bd", alpha=0.85)
    ax.axvline(g_std, color="k", ls="--", lw=1.2, label=f"global std = {g_std:.3f}")
    ax.set_xlabel("per-input local-window std  (the divisor)")
    ax.set_ylabel("count")
    ax.set_title("Distribution of input-adaptive divisors\n"
                 "spread => divisor encodes current precipitation regime")
    ax.legend(); fig.tight_layout()
    fig.savefig(osp.join(outdir, "4_divisor_hist.png"), dpi=150); plt.close(fig)

    # ---- Combined overlay CDF (the money figure) ----
    fig, ax = plt.subplots(figsize=(7, 5))
    for m, lbl, c in [(px_scalar, "pixel (scaled)", "#1f77b4"),
                       (lat_un, "latent (unnorm)", "#d62728"),
                       (lat_lw, "latent (local-window norm) [ours]", "#9467bd"),
                       (lat_no, "latent (global norm)", "#2ca02c")]:
        v = np.sort(m.values())
        if v.size:
            ax.plot(v, np.arange(1, v.size+1)/v.size, label=lbl, color=c, lw=1.6)
    ax.set_xlabel("value"); ax.set_ylabel("cumulative probability")
    ax.set_title("CDF comparison across representation spaces")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(osp.join(outdir, "0_combined_cdf.png"), dpi=150); plt.close(fig)

    with open(osp.join(outdir, "stats.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDONE -> {outdir}")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["cikm", "shanghai", "meteo"],
                   help="which dataset family")
    p.add_argument("--pixel_path", required=True)
    p.add_argument("--latent_path", required=True)
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--split", default="test")
    p.add_argument("--max_samples", type=int, default=400,
                   help="cap #sequences per pass (None/0 = all). Use a few hundred "
                        "for figures; use 0 for final paper numbers.")
    p.add_argument("--outdir", default="latent_norm_out")
    args = p.parse_args()

    name_map = {
        "cikm":     ("cikm", "cikm_latent_32"),
        "shanghai": ("shanghai", "shanghai_lr_latent_32"),
        "meteo":    ("meteo", "meteo_lr_latent_32"),
    }
    npx, nlat = name_map[args.dataset]
    ms = args.max_samples if args.max_samples > 0 else None
    run(npx, args.pixel_path, nlat, args.latent_path, args.img_size,
        args.outdir, args.split, ms)
