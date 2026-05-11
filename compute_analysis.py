"""
Computational Analysis Script
Measures: Parameters (M), FLOPs (G), GPU Memory (GB), Throughput (samples/sec)

All baselines are wrapped with InferenceWrapper so we measure exactly the same
forward path that run_alphapre_convlstm.py uses at eval time:
    radar_pred, *_ = model.predict(input, compute_loss=False)

Usage:
    python compute_analysis.py --model_name dawncast
    python compute_analysis.py --model_name all

Requirements:
    pip install fvcore --break-system-packages
"""

import torch
import torch.nn as nn
import argparse
import time
import json
from fvcore.nn import FlopCountAnalysis
from types import SimpleNamespace
from einops import rearrange

# ============================================================
# Constants — match run_alphapre_convlstm.py defaults
# ============================================================
TIN = 5
TOUT = 20
IMG_SIZE = 128
IMG_CH = 1


# ============================================================
# Wrappers
# ============================================================
class InferenceWrapper(nn.Module):
    """Wraps a model so its .predict() interface looks like a forward().

    This mirrors `_sample_batch` in run_alphapre_convlstm.py which does:
        radar_pred, *_ = model.predict(input, compute_loss=False)
    Using *_ (instead of `_`) so it works whether predict returns 2 or N values.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out, *_ = self.model.predict(x, compute_loss=False)
        return out


class E2EWrapper(nn.Module):
    """End-to-end wrapper for latent-space models: encode -> forecast -> decode.

    Use this when you want to compare a latent-space forecaster against
    pixel-space baselines on equal footing (same input/output resolution).
    """
    def __init__(self, autoencoder, forecaster, scale_factor=1.0):
        super().__init__()
        self.autoencoder = autoencoder
        self.forecaster = forecaster
        self.scale_factor = scale_factor

    def forward(self, x):
        # x: (B, T_in, C, H, W) raw pixels
        B, T, C, H, W = x.shape
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        z = self.autoencoder.encode(x).sample() * self.scale_factor
        z = rearrange(z, '(b t) c h w -> b t c h w', t=T)

        # Forecaster might expose predict() or forward()
        if hasattr(self.forecaster, 'predict'):
            z_out, *_ = self.forecaster.predict(z, compute_loss=False)
        else:
            z_out = self.forecaster(z)

        z_out = rearrange(z_out, 'b t c h w -> (b t) c h w')
        z_out = z_out / self.scale_factor
        out = self.autoencoder.decode(z_out)
        if hasattr(out, 'sample'):  # diffusers AutoencoderKL returns DecoderOutput
            out = out.sample
        out = rearrange(out, '(b t) c h w -> b t c h w', t=z_out.shape[0] // B)
        return out


# ============================================================
# Measurement helpers
# ============================================================
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(t.to(device) if isinstance(t, torch.Tensor) else t for t in x)
    return x


def measure_flops(model, dummy_input, device='cuda'):
    model = model.to(device).eval()
    dummy_input = _to_device(dummy_input, device)
    with torch.no_grad():
        return FlopCountAnalysis(model, dummy_input).total() / 1e9  # GFLOPs


def measure_gpu_memory(model, dummy_input, device='cuda', mode='peak'):
    """Measure GPU memory in GB.

    mode='peak'      -> total peak (weights + activations)  [default]
    mode='activation'-> peak minus baseline (activations only)
    """
    model = model.to(device).eval()
    dummy_input = _to_device(dummy_input, device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        _ = model(dummy_input)
    peak = torch.cuda.max_memory_allocated(device)
    used = peak if mode == 'peak' else (peak - baseline)
    return used / (1024 ** 3)


def measure_throughput(model, dummy_input, device='cuda', warmup=10, iterations=100):
    model = model.to(device).eval()
    dummy_input = _to_device(dummy_input, device)
    bs = dummy_input.shape[0] if isinstance(dummy_input, torch.Tensor) else dummy_input[0].shape[0]

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    torch.cuda.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    return (bs * iterations) / elapsed


def full_analysis(model, dummy_input, model_name, device='cuda'):
    print(f"\n{'='*50}\n  Analyzing: {model_name}\n{'='*50}")

    params = count_parameters(model)
    print(f"  Parameters: {params:.2f} M")

    try:
        flops = measure_flops(model, dummy_input, device)
        print(f"  FLOPs:      {flops:.2f} G")
    except Exception as e:
        print(f"  FLOPs:      FAILED ({e})"); flops = None

    try:
        memory = measure_gpu_memory(model, dummy_input, device)
        print(f"  GPU Memory: {memory:.2f} GB (peak total)")
    except Exception as e:
        print(f"  GPU Memory: FAILED ({e})"); memory = None

    try:
        throughput = measure_throughput(model, dummy_input, device)
        print(f"  Throughput: {throughput:.1f} samples/sec  (batch=1)")
    except Exception as e:
        print(f"  Throughput: FAILED ({e})"); throughput = None

    print(f"{'='*50}\n")
    return {
        'model': model_name,
        'params_M': round(params, 2),
        'flops_G': round(flops, 2) if flops is not None else None,
        'gpu_memory_GB': round(memory, 2) if memory is not None else None,
        'throughput_samples_per_sec': round(throughput, 1) if throughput is not None else None,
    }


# ============================================================
# Model loaders — kwargs MIRROR run_alphapre_convlstm.py exactly
# ============================================================

# --- Your model: DAWNCast ---
def load_dawncast(device='cuda'):
    """DAWNCast standalone (latent-space) — matches `--backbone dawncast` in run script."""
    from models.DAWNCast.dawncast import get_model
    kwargs = {
        "afno_blocks": 4,
        "sparsity_threshold": 0.01,
        "afno_hidden_size_factor": 3,
        "weight_scale_low": 0.1,
        "alpha_low": 1.0,
        "beta_low": 0.17,
        "freq_multiplier_low": 4.0,
        "weight_scale_high": 1.0,
        "alpha_high": 1.0,
        "beta_high": 0.17,
        "freq_multiplier_high": 4.0,
        "k_spatial": 3,
        "wave": "db6",
        "wavelet_level": 3,
        "hf_mode": 'separate',
        "img_channels": 4
    }
    model = get_model(**kwargs)
    wrapped = InferenceWrapper(model)
    # NOTE: latent-space dummy. Adjust C/H/W to match your model's defaults
    # (your run script doesn't override them, so these are whatever get_model sets internally).
    # If get_model takes pixel input and embeds internally, change to (1, TIN, 1, 128, 128).
    dummy = torch.randn(1, TIN, 4, 32, 32)
    return wrapped, dummy, "DAWNCast (latent)"


def load_dawncast_e2e(device='cuda'):
    """DAWNCast + AutoencoderKL end-to-end on pixel input.

    Use this when you want pixel-to-pixel comparison with other baselines.
    """
    from models.DAWNCast.dawncast import get_model
    from models.autoencoder_kl import AutoencoderKL

    forecaster = get_model(
        afno_blocks=4, sparsity_threshold=0.01, afno_hidden_size_factor=3,
        weight_scale_low=0.1, alpha_low=1.0, beta_low=0.17, freq_multiplier_low=4.0,
        weight_scale_high=1.0, alpha_high=1.0, beta_high=0.17, freq_multiplier_high=4.0,
        k_spatial=3, wave="db6", wavelet_level=3, hf_mode='separate',
    )
    ae = AutoencoderKL(
        in_channels=1, out_channels=1,
        down_block_types=('DownEncoderBlock2D',) * 3,
        up_block_types=('UpDecoderBlock2D',) * 3,
        block_out_channels=(128, 256, 512),
        layers_per_block=2, latent_channels=4, norm_num_groups=32,
    )
    e2e = E2EWrapper(ae, forecaster, scale_factor=1.0)  # set your real scale_factor
    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return e2e, dummy, "DAWNCast (e2e pixel)"


# --- Baselines ---
def load_alphapre(device='cuda'):
    from models.alphapre import get_model
    model = get_model(
        input_shape=(IMG_SIZE, IMG_SIZE),
        T_in=TIN, T_out=TOUT,
        img_channels=IMG_CH, dim=64,
        n_layers=3,
        pha_weight=0.01, amp_weight=0.01, anet_weight=0.1,
        spec_num=20, aweight_stop_steps=5000,
    )
    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(model), dummy, "AlphaPre"


def load_phydnet(device='cuda'):
    from models.phydnet import get_model
    model = get_model(
        in_shape=(IMG_CH, IMG_SIZE, IMG_SIZE),
        T_in=TIN, T_out=TOUT, device=device,
    )
    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(model), dummy, "PhyDNet"


def load_simvp(device='cuda'):
    from models.simvp import get_model
    model = get_model(
        in_shape=(IMG_CH, IMG_SIZE, IMG_SIZE),
        T_in=TIN, T_out=TOUT,
    )
    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(model), dummy, "SimVP"


def load_earthfarseer(device='cuda'):
    # Match run script: uses get_model with input_shape=(H, W), not the raw class
    from models.Earthfarseer.model import get_model
    model = get_model(
        input_shape=(IMG_SIZE, IMG_SIZE),
        T_in=TIN, T_out=TOUT,
        img_channels=IMG_CH,
    )
    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(model), dummy, "EarthFarseer"


def load_earthformer(device='cuda'):
    from models.earth_former import EarthFormer_xy
    model = EarthFormer_xy(in_len=TIN, out_len=TOUT, height=IMG_SIZE, width=IMG_SIZE)
    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(model), dummy, "EarthFormer"


# --- Models NOT in run_alphapre_convlstm.py — verify configs against their own run scripts ---
def load_diffcast(device='cuda'):
    """NOTE: not present in run_alphapre_convlstm.py; double-check this config
    against whichever script you actually train DiffCast with."""
    from models.phydnet import get_model as get_phydnet
    from models.diffcast import get_model as get_diffcast

    backbone = get_phydnet(in_shape=(IMG_CH, IMG_SIZE, IMG_SIZE),
                           T_in=TIN, T_out=TOUT, device=device)
    diff_model = get_diffcast(
        img_channels=IMG_CH, dim=64, dim_mults=(1, 2, 4, 8),
        T_in=TIN, T_out=TOUT, sampling_timesteps=250,
    )
    diff_model.load_backbone(backbone)

    print(f"  Backbone params: {sum(p.numel() for p in backbone.parameters())/1e6:.2f} M")
    print(f"  DiffCast total params: {sum(p.numel() for p in diff_model.parameters())/1e6:.2f} M")

    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(diff_model), dummy, "DiffCast"


def load_nowcastnet(device='cuda'):
    """NOTE: not present in run_alphapre_convlstm.py; verify this config against
    its own run script. NowcastNet uses (B, T_total, H, W, C) input format."""
    from models.nowcasting.models.nowcastnet import Net
    config = SimpleNamespace(
        device=device, worker=8, cpu_worker=8,
        input_length=TIN, total_length=TIN + TOUT,
        img_height=IMG_SIZE, img_width=IMG_SIZE, img_ch=IMG_CH,
        case_type="normal", model_name="nowcasting", batch_size=4,
        num_save_samples=10, ngf=32, evo_ic=TOUT, gen_oc=TOUT, ic_feature=320,
    )
    model = Net(config)
    # NowcastNet's NHWC (B, T_total, H, W, C) — does NOT need InferenceWrapper
    # because it doesn't expose a .predict(); kept on raw forward.
    dummy = torch.randn(1, TIN + TOUT, IMG_SIZE, IMG_SIZE, IMG_CH)
    return model, dummy, "NowcastNet"


# ============================================================
# Main
# ============================================================
MODELS = {
    'dawncast':       load_dawncast,
    'dawncast_e2e':   load_dawncast_e2e,
    'alphapre':       load_alphapre,
    'phydnet':        load_phydnet,
    'simvp':          load_simvp,
    'earthfarseer':   load_earthfarseer,
    'earthformer':    load_earthformer,
    'diffcast':       load_diffcast,
    'nowcastnet':     load_nowcastnet,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='all',
                        help=f'Model to analyze, or "all". Available: {list(MODELS.keys())}')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='compute_analysis_results.json')
    args = parser.parse_args()

    model_names = list(MODELS.keys()) if args.model_name == 'all' else [args.model_name]
    results = []

    for name in model_names:
        if name not in MODELS:
            print(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
            continue
        try:
            model, dummy, display_name = MODELS[name](args.device)
            results.append(full_analysis(model, dummy, display_name, args.device))
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[!] Failed to load/analyze {name}: {e}\n")
            results.append({'model': name, 'error': str(e)})

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'Model':<25} {'Params (M)':<12} {'FLOPs (G)':<12} {'Memory (GB)':<13} {'Throughput':<15}")
    print("-" * 77)
    for r in results:
        if 'error' in r:
            print(f"{r['model']:<25} ERROR: {r['error']}")
            continue
        params = f"{r['params_M']:.2f}" if r['params_M'] is not None else "N/A"
        flops  = f"{r['flops_G']:.2f}" if r['flops_G'] is not None else "N/A"
        mem    = f"{r['gpu_memory_GB']:.2f}" if r['gpu_memory_GB'] is not None else "N/A"
        tp     = f"{r['throughput_samples_per_sec']:.1f} s/s" if r['throughput_samples_per_sec'] is not None else "N/A"
        print(f"{r['model']:<25} {params:<12} {flops:<12} {mem:<13} {tp:<15}")