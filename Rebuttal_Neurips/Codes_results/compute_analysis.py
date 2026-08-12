"""
Computational Analysis Script
Measures: Parameters (M), FLOPs (G), GPU Memory (GB), Throughput (samples/sec)

WADEPre is wrapped with InferenceWrapper so we measure exactly the same
forward path that run_alphapre_convlstm.py uses at eval time:
    radar_pred, *_ = model.predict(input, compute_loss=False)

Usage:
    python compute_analysis.py --model_name wadepre
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


def measure_flops_diffcast(model, dummy_input, device='cuda'):
    """FLOPs for DiffCast, counted by parts.

    fvcore cannot trace DiffCast end-to-end: ContextNet.scan_ctx re-initialises
    its ConvGRU hidden states from the input shape on every call, and that init
    path produces mismatched spatial sizes under jit trace (it runs fine
    eagerly). So we count exactly the work `GaussianDiffusion.sample()` does:

        PhyDNet backbone  +  ContextNet over (T_in + T_out) frames
                          +  (T_out // T_in) frags x sampling_timesteps x UNet

    The UNet term dominates (>99%), so this is not a loose estimate.
    """
    diff = model.model if isinstance(model, InferenceWrapper) else model
    diff = diff.to(device).eval()
    frames_in = _to_device(dummy_input, device)
    B, T_in, C, H, W = frames_in.shape

    def _count(m, inp):
        with torch.no_grad():
            f = FlopCountAnalysis(m, inp)
            f.unsupported_ops_warnings(False)
            f.uncalled_modules_warnings(False)
            return f.total() / 1e9

    # --- backbone, on its own predict() path ---
    f_backbone = _count(InferenceWrapper(diff.backbone_net).to(device).eval(), frames_in)

    # --- context net: run the scan eagerly (works) to harvest the real ctx
    #     pyramid, then trace a single frame with the state already in place ---
    with torch.no_grad():
        backbone_out, _ = diff.backbone_net.predict(frames_in)
        ctx_frames = torch.cat((diff.normalize(frames_in),
                                diff.normalize(backbone_out)), dim=1)
        global_ctx, _ = diff.ctx_net.scan_ctx(ctx_frames)
        diff.ctx_net.init_state((B, C, H, W), device)

    class _CtxStep(nn.Module):
        def __init__(self, net):
            super().__init__(); self.net = net

        def forward(self, frame):
            return self.net.forward(frame)[-1]

    f_ctx = _count(_CtxStep(diff.ctx_net).to(device).eval(), ctx_frames[:, 0]) * ctx_frames.shape[1]

    # --- one denoiser evaluation, called exactly as ddim_sample calls it ---
    class _UnetStep(nn.Module):
        def __init__(self, unet, ctx):
            super().__init__(); self.unet = unet; self.ctx = ctx

        def forward(self, x, time, cond, idx):
            return self.unet(x, time, cond=cond, ctx=self.ctx, idx=idx)

    x = torch.randn(B, T_in, C, H, W, device=device)
    unet_args = (x, torch.zeros(B, device=device, dtype=torch.long), torch.zeros_like(x),
                 torch.zeros(B, device=device, dtype=torch.long))
    f_unet = _count(_UnetStep(diff.model, global_ctx).to(device).eval(), unet_args)

    n_frag = diff.T_out // diff.T_in
    f_sampling = n_frag * diff.sampling_timesteps * f_unet

    print(f"    backbone            : {f_backbone:10.2f} G")
    print(f"    context net (x{ctx_frames.shape[1]})   : {f_ctx:10.2f} G")
    print(f"    UNet, 1 DDIM step   : {f_unet:10.2f} G")
    print(f"    x {n_frag} frags x {diff.sampling_timesteps} steps : {f_sampling:10.2f} G")
    return f_backbone + f_ctx + f_sampling


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


def full_analysis(model, dummy_input, model_name, device='cuda',
                  warmup=10, iterations=100, flops_fn=None):
    print(f"\n{'='*50}\n  Analyzing: {model_name}\n{'='*50}")
    flops_fn = flops_fn or measure_flops

    params = count_parameters(model)
    print(f"  Parameters: {params:.2f} M")

    try:
        flops = flops_fn(model, dummy_input, device)
        print(f"  FLOPs:      {flops:.2f} G")
    except Exception as e:
        print(f"  FLOPs:      FAILED ({e})"); flops = None

    try:
        memory = measure_gpu_memory(model, dummy_input, device)
        print(f"  GPU Memory: {memory:.2f} GB (peak total)")
    except Exception as e:
        print(f"  GPU Memory: FAILED ({e})"); memory = None

    try:
        throughput = measure_throughput(model, dummy_input, device,
                                        warmup=warmup, iterations=iterations)
        print(f"  Throughput: {throughput:.4g} samples/sec  "
              f"({1/throughput:.3g} s/sample, batch=1, {iterations} iters)")
    except Exception as e:
        print(f"  Throughput: FAILED ({e})"); throughput = None

    print(f"{'='*50}\n")
    return {
        'model': model_name,
        'params_M': round(params, 2),
        'flops_G': round(flops, 2) if flops is not None else None,
        'gpu_memory_GB': round(memory, 2) if memory is not None else None,
        'throughput_samples_per_sec': float(f"{throughput:.4g}") if throughput is not None else None,
        'latency_sec_per_sample': float(f"{1/throughput:.4g}") if throughput else None,
    }


# ============================================================
# Model loaders — kwargs MIRROR run_alphapre_convlstm.py exactly
# ============================================================
def load_wadepre(device='cuda'):
    """WADEPre — matches `--backbone wadepre` in run_alphapre_convlstm.py.

    Same-length model: predict() with compute_loss=False rolls out
    autoregressively ceil(T_out / T_in) times (4x for 5 -> 20), so the numbers
    below cover the full 20-frame forecast, not a single block.
    """
    from models.WADEPre.wadepre import get_model
    model = get_model(
        input_shape=(IMG_SIZE, IMG_SIZE),
        T_in=TIN, T_out=TOUT,
        img_channels=IMG_CH,
    )
    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(model), dummy, "WADEPre"


def load_diffcast(device='cuda'):
    """DiffCast (PhyDNet backbone) — mirrors run_diffcast.py's build block.

    run_diffcast.py defaults to --backbone phydnet and always calls
    get_model(img_channel, 64, (1,2,4,8), T_in, T_out, 1000,
              sampling_timesteps=250) when --use_diff is set.

    Inference is a 250-step DDIM loop over the UNet, so one forward is ~250
    denoiser evaluations. See MEASURE_OPTS for the reduced timing budget.
    """
    from models.phydnet import get_model as get_phydnet
    from models.diffcast import get_model as get_diffcast

    backbone = get_phydnet(in_shape=(IMG_CH, IMG_SIZE, IMG_SIZE),
                           T_in=TIN, T_out=TOUT, device=device)
    diff_model = get_diffcast(
        IMG_CH, 64, (1, 2, 4, 8), TIN, TOUT, 1000, sampling_timesteps=250,
    )
    diff_model.load_backbone(backbone)

    print(f"  Backbone (PhyDNet) params: {sum(p.numel() for p in backbone.parameters())/1e6:.2f} M")
    print(f"  DiffCast total params:     {sum(p.numel() for p in diff_model.parameters())/1e6:.2f} M")

    dummy = torch.randn(1, TIN, IMG_CH, IMG_SIZE, IMG_SIZE)
    return InferenceWrapper(diff_model), dummy, "DiffCast"


# ============================================================
# Main
# ============================================================
MODELS = {
    'wadepre':        load_wadepre,
    'diffcast':       load_diffcast,
}

# Per-model overrides for full_analysis. Only needed for models where the
# default 10 warmup + 100 timed iterations is impractical.
MEASURE_OPTS = {
    # One DiffCast forward = 250 DDIM steps, so 110 forwards would take hours
    # and fvcore would have to trace the whole unrolled sampling loop.
    'diffcast': dict(warmup=1, iterations=5, flops_fn=measure_flops_diffcast),
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
            results.append(full_analysis(model, dummy, display_name, args.device,
                                         **MEASURE_OPTS.get(name, {})))
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
        tp     = f"{r['throughput_samples_per_sec']:.4g} s/s" if r['throughput_samples_per_sec'] is not None else "N/A"
        print(f"{r['model']:<25} {params:<12} {flops:<12} {mem:<13} {tp:<15}")
