"""
Computational Analysis Script
Measures: Parameters (M), FLOPs (G), GPU Memory (GB)

Usage:
    python compute_analysis.py --model_name lastocast

Requirements:
    pip install fvcore --break-system-packages
"""

import torch
import torch.nn as nn
import argparse
import time
import json
import os
from fvcore.nn import FlopCountAnalysis

# ============================================================
# Helper functions
# ============================================================
TIN = 5
TOUT=20 

def count_parameters(model):
    """Count total learnable parameters in millions."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6

def measure_flops(model, dummy_input, device='cuda'):
    """Measure FLOPs in GFLOPs using fvcore."""
    model = model.to(device).eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.to(device)
    elif isinstance(dummy_input, (list, tuple)):
        dummy_input = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in dummy_input)
    
    with torch.no_grad():
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
    return total_flops / 1e9  # GFLOPs

def measure_gpu_memory(model, dummy_input, device='cuda'):
    """Measure peak GPU memory during forward pass in GB."""
    model = model.to(device).eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.to(device)
    elif isinstance(dummy_input, (list, tuple)):
        dummy_input = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in dummy_input)
    
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Measure baseline memory (model only)
    baseline_mem = torch.cuda.memory_allocated(device)
    
    with torch.no_grad():
        _ = model(dummy_input)
    
    peak_mem = torch.cuda.max_memory_allocated(device)
    return peak_mem / (1024 ** 3)  # GB

def measure_throughput(model, dummy_input, device='cuda', warmup=10, iterations=100):
    """Measure inference throughput in samples/sec."""
    model = model.to(device).eval()
    if isinstance(dummy_input, torch.Tensor):
        dummy_input = dummy_input.to(device)
        batch_size = dummy_input.shape[0]
    elif isinstance(dummy_input, (list, tuple)):
        dummy_input = tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in dummy_input)
        batch_size = dummy_input[0].shape[0]
    
    # Warmup runs (GPU needs to reach steady state)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    torch.cuda.synchronize()
    
    # Timed runs
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    throughput = (batch_size * iterations) / elapsed
    return throughput

def full_analysis(model, dummy_input, model_name, device='cuda'):
    """Run all measurements and return a dict."""
    print(f"\n{'='*50}")
    print(f"  Analyzing: {model_name}")
    print(f"{'='*50}")
    
    # Parameters
    params = count_parameters(model)
    print(f"  Parameters: {params:.2f} M")
    
    # FLOPs
    try:
        flops = measure_flops(model, dummy_input, device)
        print(f"  FLOPs:      {flops:.2f} G")
    except Exception as e:
        print(f"  FLOPs:      FAILED ({e})")
        flops = None
    
    # GPU Memory
    try:
        memory = measure_gpu_memory(model, dummy_input, device)
        print(f"  GPU Memory: {memory:.2f} GB")
    except Exception as e:
        print(f"  GPU Memory: FAILED ({e})")
        memory = None
    
    # Throughput
    try:
        throughput = measure_throughput(model, dummy_input, device)
        print(f"  Throughput: {throughput:.1f} samples/sec")
    except Exception as e:
        print(f"  Throughput: FAILED ({e})")
        throughput = None
    
    print(f"{'='*50}\n")
    
    return {
        'model': model_name,
        'params_M': round(params, 2),
        'flops_G': round(flops, 2) if flops else None,
        'gpu_memory_GB': round(memory, 2) if memory else None,
        'throughput_samples_per_sec': round(throughput, 1) if throughput else None,
    }


# ============================================================
# Model loaders — ADD YOUR BASELINES HERE
# ============================================================

def load_lastocast(device='cuda'):
    """Load your model."""
    # Adjust the import path as needed
    from models.Lastocast.lastocast import get_model
    
    model = get_model(
        weight_scale=1.5, alpha=1.0, beta=1.0, freq_multiplier=1.0,size_factor=1.0,
        total_steps=50000, const_ratio=0.5,
        img_channels=4, dim=64,
        T_in= TIN, T_out=TOUT,
        input_shape=(32, 32)
    )
    # Input shape: (B, T_in, C, H, W) in latent space
    dummy = torch.randn(1, 5, 4, 32, 32)
    return model, dummy, "LASTOCast (Ours)"


# ============================================================
# Template for adding baselines — copy and modify
# ============================================================

def load_alphapre(device='cuda'):
    from models.alphapre import get_model
    model = get_model(input_shape = (128,128), T_in=TIN, T_out=TOUT, img_channels=1, dim=64,
                      n_layers=3, pha_weight=0.01, amp_weight=0.01, anet_weight=0.1, spec_num=20,
                      aweight_stop_steps = 5000)  # use same config as your experiments
    dummy = torch.randn(1, 5, 1, 128, 128)  # match input shape
    return model, dummy, "AlphaPre"

def load_diffcast(device='cuda'):
    
    from models.phydnet import get_model
    model = get_model(in_shape=(1,128,128), T_in=TIN, T_out=TOUT, device=device)
    from models.diffcast import get_model
    diff_model = get_model(img_channels=1, dim=64, dim_mults=(1,2,4,8), T_in=TIN, T_out=TOUT, sampling_timesteps=250)
    diff_model.load_backbone(model)
    dummy = torch.randn(1, 5, 1, 128, 128)
    return diff_model, dummy, "DiffCast"

# def load_nowcastnet(device='cuda'):
#     from models.nowcastnet import get_model
#     model = get_model(...)
#     dummy = torch.randn(1, 5, 4, 32, 32)
#     return model, dummy, "NowcastNet"

# def load_earthfarseer(device='cuda'):
#     from models.earthfarseer import get_model
#     model = get_model(...)
#     dummy = torch.randn(1, 5, 4, 32, 32)
#     return model, dummy, "EarthFarseer"

def load_earthformer(device='cuda'):
    from models.earth_former import EarthFormer_xy
    model = EarthFormer_xy(in_len=TIN, out_len=TOUT, height=128, width=128)
    dummy = torch.randn(1, 5, 1, 128, 128)
    return model, dummy, "EarthFormer"

def load_phydnet(device='cuda'):
    from models.phydnet import get_model
    model = get_model(in_shape=(1,128,128), T_in=TIN, T_out=TOUT, device=device)
    dummy = torch.randn(1, 5, 1, 128, 128)
    return model, dummy, "PhyDNet"


# ============================================================
# Main
# ============================================================

MODELS = {
    'lastocast': load_lastocast,
    'alphapre': load_alphapre,
    'diffcast': load_diffcast,
    # 'nowcastnet': load_nowcastnet,
    # 'earthfarseer': load_earthfarseer,
    'earthformer': load_earthformer,
    'phydnet': load_phydnet,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='all',
                        help='Model to analyze, or "all" for all models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='compute_analysis_results.json')
    args = parser.parse_args()

    results = []

    if args.model_name == 'all':
        model_names = list(MODELS.keys())
    else:
        model_names = [args.model_name]

    for name in model_names:
        if name not in MODELS:
            print(f"Unknown model: {name}. Available: {list(MODELS.keys())}")
            continue
        
        loader = MODELS[name]
        model, dummy, display_name = loader(args.device)
        result = full_analysis(model, dummy, display_name, args.device)
        results.append(result)
        
        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print summary table
    print(f"\n{'Model':<25} {'Params (M)':<12} {'FLOPs (G)':<12} {'Memory (GB)':<12} {'Throughput':<15}")
    print("-" * 76)
    for r in results:
        params = f"{r['params_M']:.2f}" if r['params_M'] else "N/A"
        flops = f"{r['flops_G']:.2f}" if r['flops_G'] else "N/A"
        mem = f"{r['gpu_memory_GB']:.2f}" if r['gpu_memory_GB'] else "N/A"
        tp = f"{r['throughput_samples_per_sec']:.1f} s/s" if r['throughput_samples_per_sec'] else "N/A"
        print(f"{r['model']:<25} {params:<12} {flops:<12} {mem:<12} {tp:<15}")