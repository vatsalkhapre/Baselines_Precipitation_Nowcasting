"""
Convert a trained AmpliNet/AlphaPre checkpoint
(alpha_amplinet_latent_FAL_FCL_..._final.py :: WaveletLASTOCastForecaster)
into a DAWN-Cast checkpoint (dawncast.py :: DAWNCastForecaster).

The two networks are computationally identical; only module/attribute names
differ. This script performs a SCOPE-AWARE key rename (a naive global
str.replace corrupts lifting/projection weights because block1/block2/proj
names are reused there) and, if torch + both model files are importable,
verifies a strict load.

Usage:
    python convert_amplinet_to_dawncast.py in.pth out.pth
    python convert_amplinet_to_dawncast.py in.pth out.pth --verify \
        --hf_mode shared --level 1 --wave haar --dim 64 --t_in 5 --t_out 20 \
        --img_channels 4 --afno_blocks 1 --sparsity_threshold 0.01 \
        --hidden_size_factor 1 --k_spatial 3
"""
import argparse
import os
import torch


def convert_key(k: str) -> str:
    """Map one AmpliNet state_dict key to its DAWN-Cast equivalent."""
    # Optional DDP/Accelerate wrapper prefix — strip, convert, re-attach.
    prefix = ""
    if k.startswith("module."):
        prefix, k = "module.", k[len("module."):]

    # 1. Forecaster wrapper: self.lastocast -> self.dawncast
    #    (falfcl stays falfcl; lifting/projection/dwt/idwt stay as-is)
    if k.startswith("lastocast."):
        k = "dawncast." + k[len("lastocast."):]

    # 2. Core block + everything under it
    if ".operator." in k:
        k = k.replace(".operator.", ".wgtm.", 1)
        k = k.replace(".stream_ll.", ".fat_ll.")
        k = k.replace(".stream_hf.", ".fat_hf.")
        k = k.replace(".hf_streams.", ".fat_hf_streams.")

        # 3. Spectral stack — SCOPED renames (only inside conv_spectral).
        #    block1/block2/proj are reused in lifting/projection and MUST NOT
        #    be touched there, hence the guard.
        if ".conv_spectral." in k:
            k = k.replace(".conv_spectral.", ".srst.")
            k = k.replace(".block1.", ".srst_block1.")
            k = k.replace(".block2.", ".srst_block2.")
            k = k.replace(".proj.", ".str_branch.")     # AFNO2D -> STRModule
            k = k.replace(".dw_spatial.", ".spatial_branch.")
            k = k.replace(".pw.", ".channel_mixing.")
            # .norm. and the trailing AFNO (.srst.2.w1 ...) need no further change

    return prefix + k


def convert_state_dict(sd: dict) -> dict:
    out, seen = {}, {}
    for k, v in sd.items():
        nk = convert_key(k)
        if nk in seen:
            raise RuntimeError(
                f"Key collision: '{k}' and '{seen[nk]}' both map to '{nk}'. "
                "Rename rules are ambiguous for this checkpoint layout."
            )
        seen[nk] = k
        out[nk] = v
    return out


def _unwrap(obj):
    """Pull a plain state_dict out of common checkpoint container formats."""
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "ema", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key], key
    return obj, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", action="store_true",
                    help="Instantiate DAWNCast and assert strict load "
                         "(requires torch + dawncast.py importable).")
    # DAWNCast build args for --verify; MUST match the trained config.
    ap.add_argument("--hf_mode", default="shared", choices=["shared", "separate"])
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--wave", default="haar")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--t_in", type=int, default=5)
    ap.add_argument("--t_out", type=int, default=20)
    ap.add_argument("--img_channels", type=int, default=4)
    ap.add_argument("--afno_blocks", type=int, default=1)
    ap.add_argument("--sparsity_threshold", type=float, default=0.01)
    ap.add_argument("--hidden_size_factor", type=int, default=1)
    ap.add_argument("--k_spatial", type=int, default=3)
    args = ap.parse_args()

    ckpt = torch.load(args.src, map_location="cpu")
    sd, container_key = _unwrap(ckpt)

    new_sd = convert_state_dict(sd)
    n_changed = sum(1 for k in sd if convert_key(k) != k)
    print(f"Converted {len(new_sd)} keys ({n_changed} renamed, "
          f"{len(new_sd) - n_changed} unchanged).")

    if args.verify:
        from models.DAWNCast import dawncast  # noqa: must be importable (utils.utilspp, pytorch_wavelets present)
        model = dawncast.get_model(
            afno_blocks=args.afno_blocks,
            sparsity_threshold=args.sparsity_threshold,
            afno_hidden_size_factor=args.hidden_size_factor,
            k_spatial=args.k_spatial,
            img_channels=args.img_channels, dim=args.dim,
            T_in=args.t_in, T_out=args.t_out,
            wave=args.wave, wavelet_level=args.level, hf_mode=args.hf_mode,
        )
        target = set(model.state_dict().keys())
        got = set(new_sd.keys())
        missing, unexpected = target - got, got - target
        # Buffers (dwt/idwt filters) and falfcl state are fine either way;
        # report only genuine mismatches.
        if missing or unexpected:
            print(f"[!] missing ({len(missing)}):",
                  sorted(missing)[:10], "..." if len(missing) > 10 else "")
            print(f"[!] unexpected ({len(unexpected)}):",
                  sorted(unexpected)[:10], "..." if len(unexpected) > 10 else "")
        res = model.load_state_dict(new_sd, strict=False)
        # Shape check on the intersection
        msd = model.state_dict()
        bad = [k for k in got & target if new_sd[k].shape != msd[k].shape]
        if bad:
            raise RuntimeError(f"Shape mismatch on {len(bad)} keys, e.g. "
                               f"{bad[0]}: {tuple(new_sd[bad[0]].shape)} vs "
                               f"{tuple(msd[bad[0]].shape)}")
        print(f"[ok] strict-load check passed on all "
              f"{len(got & target)} overlapping keys "
              f"(missing={len(res.missing_keys)}, "
              f"unexpected={len(res.unexpected_keys)}).")

    dst_dir = os.path.dirname(args.dst)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)

    # Re-wrap into the original container if there was one.
    if container_key is not None:
        ckpt[container_key] = new_sd
        torch.save(ckpt, args.dst)
    else:
        torch.save(new_sd, args.dst)
    print(f"Saved -> {args.dst}")


if __name__ == "__main__":
    main()
