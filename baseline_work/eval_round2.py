#!/usr/bin/env python
"""Test-evaluate a finished round-2 / DiffCast cell from its own params.yaml.

Cells are trained with --valid (which tracks the best-CSI checkpoint) but never
run a TEST evaluation, so their numbers exist only as validation. The table must
not mix validation and test numbers, so every finished cell gets an explicit
inference-only test pass here, using the SAME evaluator as every other baseline.

Args are read back from the run's own params.yaml so the model is rebuilt exactly
as trained -- nothing is assumed.
"""
import os, subprocess, sys, yaml
REPO='/home/vatsal/NWM/Baselines_Precipitation_Nowcasting'
PY='/home/vatsal/miniconda3/envs/earthformer/bin/python'
CSV='/home/vatsal/Dataserver2/Neurips/csv_files/models_falfcl.csv'

def main(expdir, note, gpu):
    d=f'{REPO}/Exps/{expdir}/{note}'
    c=yaml.safe_load(open(f'{d}/params.yaml'))
    ck=f'{d}/checkpoints/ckpt-best.pt'
    assert os.path.exists(ck), ck
    is_diff = bool(c.get('use_diff'))
    runner = 'run_diffcast_falfcl.py' if is_diff else 'run_baselines.py'
    cmd=[PY,f'{REPO}/{runner}','--exp_dir',expdir,'--exp_note',note,
         '--backbone',str(c['backbone']),'--dataset',str(c['dataset']),
         '--img_size',str(c.get('img_size',128)),'--frames_in',str(c.get('frames_in',5)),
         '--frames_out',str(c.get('frames_out',20)),'--batch_size',str(c.get('batch_size',4)),
         '--seq_len','25','--epochs',str(c.get('epochs',50)),
         '--eval','--ckpt_milestone',ck,'--num_workers','8','--wandb_state','offline',
         '--wandb_project_name','ICLR26_FACL_runs','--run_name',f'testeval_{note}']
    if is_diff:
        cmd.append('--use_diff')
    else:
        cmd += ['--results_csv',CSV]
        for k,flag in [('drop_path_rate','--drop_path_rate'),('embed_dim','--embed_dim'),
                       ('skip_connection','--skip_connection'),('lr','--lr'),
                       ('refine_hidden_dim','--refine_hidden_dim')]:
            if c.get(k) is not None: cmd += [flag,str(c[k])]
        for k,flag in [('depths','--depths'),('num_heads','--num_heads')]:
            if c.get(k) is not None: cmd += [flag,str(c[k])]
    print(' '.join(cmd),flush=True)
    return subprocess.call(cmd,cwd=REPO,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)))

if __name__=='__main__':
    sys.exit(main(sys.argv[1],sys.argv[2],sys.argv[3]))
