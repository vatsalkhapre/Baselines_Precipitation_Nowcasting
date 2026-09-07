#!/usr/bin/env python
"""Evaluate the pretrained DAWN-Cast checkpoints for the results table.

Inference only -- these are the owner's pretrained pixel-space DAWN-Cast models
and are NOT retrained or modified. Each run's own params.yaml supplies its
hyperparameters (wavelet, Gabor params, spectral blocks differ per dataset), so
nothing is assumed or shared between them.

Backbone mapping was established EMPIRICALLY, not from the config string: each
checkpoint was strict-loaded against both models/DAWNCast/dawncast.py and
dawncast_old.py and the module giving 0 missing / 0 unexpected keys was taken.
This matters because the SEVIR run records backbone 'dawncast' (lowercase),
which the runner does not accept -- it only knows 'DAWNCast' and 'DAWNCast_old'.
Three of the five resulting parameter counts independently match the values
recorded in Parameter_budget.csv, confirming the mapping.

Outputs go to a fresh Exps/dawncast_eval/<tag> directory; the pretrained
checkpoint directories are read-only here and are never written to.
"""
import os, subprocess, sys, yaml

B   = '/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Best_ckpt_pixel'
REPO= '/home/vatsal/NWM/Baselines_Precipitation_Nowcasting'
PY  = '/home/vatsal/miniconda3/envs/earthformer/bin/python'
CSV = '/home/vatsal/Dataserver2/Neurips/csv_files/dawncast_pretrained_eval.csv'

# tag -> (path under B, backbone arg to pass)
RUNS = [
    ('dawncast_sevir',      'SEVIR/dawncast_sevir_pixel',                   'DAWNCast'),
    ('dawncast_cikm',       'CIKM/CIKM_pixel_flow22.74_fhigh95.56',         'DAWNCast_old'),
    ('dawncast_shanghai_a', 'Shanghai/Shanghai_pixel_flow1.09_fhigh0.14',   'DAWNCast'),
    ('dawncast_shanghai_b', 'Shanghai/Shanghai_pixel_flow1.09_fhigh4.43',   'DAWNCast'),
    ('dawncast_meteo',      'Meteonet/Meteonet_pixel_flow1.09_fhigh1.12',   'DAWNCast'),
]

def build(tag, sub, backbone):
    c = yaml.safe_load(open(f'{B}/{sub}/params.yaml'))
    ck = f'{B}/{sub}/checkpoints/ckpt-best.pt'
    assert os.path.exists(ck), ck
    a = lambda k, d=None: c.get(k, d)
    return [PY, f'{REPO}/run_baselines.py',
        '--exp_dir','dawncast_eval','--exp_note',tag,
        '--backbone',backbone,'--dataset',str(a('dataset')),
        '--img_size',str(a('img_size')),'--frames_in',str(a('frames_in')),
        '--frames_out',str(a('frames_out')),'--batch_size',str(a('batch_size',4)),
        '--seq_len','25','--img_channel',str(a('img_channel',1)),
        # DAWN-Cast hyperparameters, taken verbatim from this run's params.yaml
        '--wave',str(a('wave')),'--wavelet_level',str(a('wavelet_level')),
        '--hf_mode',str(a('hf_mode')),'--conv_kernel',str(a('conv_kernel')),
        '--sparsity_threshold',str(a('sparsity_threshold')),
        '--spectral_blocks',str(a('spectral_blocks')),
        '--spectral_hidden_size_factor',str(a('spectral_hidden_size_factor')),
        '--hidden_dim',str(a('hidden_dim')),'--size_factor',str(a('size_factor',1.0)),
        '--weight_scale_low',str(a('weight_scale_low')),'--alpha_low',str(a('alpha_low')),
        '--beta_low',str(a('beta_low')),'--freq_multiplier_low',str(a('freq_multiplier_low')),
        '--weight_scale_high',str(a('weight_scale_high')),'--alpha_high',str(a('alpha_high')),
        '--beta_high',str(a('beta_high')),'--freq_multiplier_high',str(a('freq_multiplier_high')),
        '--layers',str(a('layers',3)),
        '--eval','--ckpt_milestone',ck,
        '--num_workers','8','--wandb_state','offline',
        '--wandb_project_name','ICLR26_FACL_runs','--run_name',f'eval_{tag}',
        '--results_csv',CSV]

if __name__ == '__main__':
    which = sys.argv[1]; gpu = sys.argv[2]
    tag, sub, bb = next(r for r in RUNS if r[0] == which)
    cmd = build(tag, sub, bb)
    print(' '.join(cmd), flush=True)
    sys.exit(subprocess.call(cmd, cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)))
