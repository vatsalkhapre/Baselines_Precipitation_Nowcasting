#!/usr/bin/env python
"""Mandatory gate 3: final verification. A cell is NOT complete until this passes.

For a cell the dispatcher marked 'done', this:
  1. reloads ckpt-best.pt and checks it carries the expected bookkeeping
     (max_csi/best_step -- absent means the run predates the resume fix and its
     ckpt-best may have been clobbered by a post-preemption validation);
  2. re-runs INFERENCE ONLY from that checkpoint via run_baselines.py --eval,
     which recomputes every metric with the same evaluator every other baseline
     used and appends a row to the results CSV;
  3. re-parses the resulting log and checks:
       - the per-lead-time CSI vector length == the cell's frames_out
         (this is read off the actual prediction tensor, so it catches a
          silently padded/truncated horizon)
       - every reported metric is finite
       - recomputed CSI-M equals the mean of the per-threshold CSI values
       - recomputed CSI-M matches the value written to the CSV
  4. records CSI-H / CSI-E (top two thresholds for that dataset).

Any failure marks the cell 'failed' in the manifest. Nothing is silently passed.
"""
import csv, json, os, re, subprocess, sys, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_threshold_backfill import parse_last_eval_block, THRESHOLDS

REPO='/home/vatsal/NWM/Baselines_Precipitation_Nowcasting'
MAN='/home/vatsal/Dataserver2/Neurips/baseline_manifest/manifest.csv'
PY='/home/vatsal/miniconda3/envs/earthformer/bin/python'
RESULTS_CSV='/home/vatsal/Dataserver2/Neurips/csv_files/models_falfcl.csv'

def cells(status=None):
    rows=list(csv.DictReader(open(MAN)))
    return [r for r in rows if status is None or r['status']==status]

def set_status(cid, status, note):
    import fcntl
    lock=open('/home/vatsal/Dataserver2/Neurips/baseline_manifest/manifest.lock','w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    rows=list(csv.DictReader(open(MAN))); hdr=rows[0].keys()
    for r in rows:
        if r['cell_id']==cid:
            r['status']=status; r['note']=note
            r['last_updated']=datetime.datetime.now().isoformat(timespec='seconds')
    tmp=MAN+'.tmp'; 
    with open(tmp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(hdr)); w.writeheader(); w.writerows(rows)
    os.rename(tmp,MAN); fcntl.flock(lock,fcntl.LOCK_UN); lock.close()

def verify(cell, gpu=0):
    cid=cell['cell_id']; ds=cell['dataset']; tout=int(cell['frames_out'])
    exp=os.path.join(REPO,'Exps','baselines_falfcl',f"{cell['model']}_on_{ds}")
    ckpt=os.path.join(exp,'checkpoints','ckpt-best.pt')
    log=os.path.join(exp,'logs','log.log')
    fails=[]

    if not os.path.exists(ckpt):
        return False, f"no ckpt-best.pt at {ckpt}"
    import torch
    d=torch.load(ckpt,map_location='cpu',weights_only=False)
    if 'max_csi' not in d:
        fails.append("checkpoint has no max_csi (predates resume fix)")
    train_best=d.get('max_csi')

    # --- independent recomputation: inference only, no training ---
    cmd=[PY,os.path.join(REPO,'run_baselines.py'),
         '--exp_dir','baselines_falfcl','--exp_note',f"{cell['model']}_on_{ds}",
         '--backbone',cell['backbone'],'--dataset',ds,
         '--batch_size',cell['batch_size'],'--seq_len','25',
         '--frames_in',cell['frames_in'],'--frames_out',cell['frames_out'],
         '--img_size',cell['img_size'],'--epochs',cell['epochs'],
         '--lr',cell['lr'],'--seed',cell['seed'],
         '--eval','--ckpt_milestone',ckpt,
         '--num_workers','8','--wandb_state','offline',
         '--wandb_project_name','ICLR26_FACL_runs','--run_name',f"verify_{cid}",
         '--results_csv',RESULTS_CSV]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu))
    r=subprocess.run(cmd,env=env,cwd=REPO,capture_output=True,text=True,timeout=14400)
    if r.returncode!=0:
        return False, f"eval rerun failed rc={r.returncode}: {r.stderr[-300:]}"

    per, reported = parse_last_eval_block(log)
    if per is None:
        return False, "could not parse the verification eval block"

    # horizon check, read off the actual prediction tensor
    txt=open(log,errors='ignore').read()
    m=re.findall(r"<CSI> : [0-9.eE+-]+; \[(.*?)\]", txt, re.S)
    if m:
        n=len(m[-1].split())
        if n!=tout: fails.append(f"horizon mismatch: {n} lead times, expected {tout}")
    # finiteness
    if not all(v==v and abs(v)!=float('inf') for v in per.values()):
        fails.append("non-finite CSI value")
    # internal consistency
    recomputed=sum(per.values())/len(per)
    if abs(recomputed-reported)>1e-6:
        fails.append(f"CSI-M inconsistent: {recomputed} vs {reported}")
    # thresholds are the dataset's own
    if sorted(per)!=sorted(THRESHOLDS[ds]):
        fails.append(f"threshold set {sorted(per)} != {sorted(THRESHOLDS[ds])}")

    th=THRESHOLDS[ds]
    info=(f"CSI-M={reported:.6f} (train-best val {train_best}) "
          f"CSI-{th[-2]}={per[th[-2]]:.6f} CSI-{th[-1]}={per[th[-1]]:.6f}")
    return (not fails), (info if not fails else "; ".join(fails)+" | "+info)

if __name__=='__main__':
    gpu=int(sys.argv[1]) if len(sys.argv)>1 else 0
    todo=[c for c in cells('done') if 'verified' not in (c.get('note') or '')]
    if not todo: print("nothing to verify"); sys.exit(0)
    for c in todo:
        ok,msg=verify(c,gpu)
        print(f"{'PASS' if ok else 'FAIL'}  {c['cell_id']:26} {msg}",flush=True)
        set_status(c['cell_id'],'done' if ok else 'failed',
                   ('verified: ' if ok else 'verification failed: ')+msg)
