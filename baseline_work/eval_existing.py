#!/usr/bin/env python
"""Run a TEST evaluation on an already-trained run that only ever got validated.

Four runs in Models_falfcl have checkpoints and validation numbers but no test
evaluation (a test row is only produced when the runner is invoked with --eval).
The results table must not mix validation numbers with test numbers, so those
four are evaluated here with the same evaluator every other cell used.

Inference only. Uses the backbone recorded in each run's own params.yaml -- the
weights are evaluated exactly as they were trained, not under a newer variant.
"""
import os, subprocess, sys, yaml
M='/home/vatsal/Dataserver2/Neurips/Models_falfcl'
REPO='/home/vatsal/NWM/Baselines_Precipitation_Nowcasting'
PY='/home/vatsal/miniconda3/envs/earthformer/bin/python'
CSV='/home/vatsal/Dataserver2/Neurips/csv_files/models_falfcl.csv'
RUNS={
 'earthformer_on_meteonet':'meteonet_falfcl/earthformer_on_meteonet',
 'simvp_on_meteo':         'meteonet_falfcl/simvp_on_meteo',
 'simvp_on_sevir':         'sevir_falfcl/simvp_on_sevir',
 'trajgru_on_sevir':       'sevir_falfcl/trajgru_on_sevir',
}
def main(tag, gpu):
    sub=RUNS[tag]; d=f'{M}/{sub}'
    c=yaml.safe_load(open(f'{d}/params.yaml'))
    ck=f'{d}/checkpoints/ckpt-best.pt'; assert os.path.exists(ck), ck
    dsdir=sub.split('/')[0]
    os.makedirs(f'{REPO}/Exps', exist_ok=True)
    link=f'{REPO}/Exps/eval_{dsdir}'
    if not os.path.islink(link): os.symlink(f'{M}/{dsdir}', link)
    bb = str(c['backbone'])
    # SimVP checkpoints predate the N_T 6->4 change in simvp_falfcl/simvp_iter.py
    if bb == 'simvp_falfcl': bb = 'simvp_falfcl_nt6'
    cmd=[PY,f'{REPO}/run_baselines.py','--exp_dir',f'eval_{dsdir}','--exp_note',os.path.basename(sub),
        '--backbone',bb,'--dataset',str(c['dataset']),
        '--img_size',str(c.get('img_size',128)),'--frames_in',str(c.get('frames_in',5)),
        '--frames_out',str(c.get('frames_out',20)),'--batch_size','4','--seq_len','25',
        '--layers',str(c.get('layers',3)),'--hidden_dim',str(c.get('hidden_dim',64)),
        '--eval','--ckpt_milestone',ck,'--num_workers','8','--wandb_state','offline',
        '--wandb_project_name','ICLR26_FACL_runs','--run_name',f'testeval_{tag}',
        '--results_csv',CSV]
    print(' '.join(cmd),flush=True)
    return subprocess.call(cmd,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu))
if __name__=='__main__': sys.exit(main(sys.argv[1],sys.argv[2]))
