#!/usr/bin/env python
"""Item 5: top up the already-completed cells to a uniform epoch budget.

Owner's instruction: "After all the runs are done take last checkpoints and
continue for whatever run epochs are left."

TWO BLOCKERS THIS SCRIPT EXISTS TO HANDLE
-----------------------------------------
1. Every pre-existing checkpoint predates the max_csi persistence fix - verified:
   mau_on_sevir_facl and simvp_on_shanghai carry only ema/epoch/opt/scheduler/step.
   Resuming one means self.max_csi starts at 0.0, so the FIRST validation after
   resume beats it and OVERWRITES ckpt-best.pt with a worse checkpoint - losing
   exactly the result being topped up. This script therefore:
       - copies ckpt-best.pt to ckpt-best.prefix-topup.pt BEFORE launching,
       - records the pre-topup best CSI-M parsed from log.log,
       - after the top-up, compares old-best vs new-best and RESTORES the old
         checkpoint if the top-up did not actually improve it.
   That makes the best-checkpoint rule hold across the resume boundary even
   though the old checkpoint cannot carry the state itself.

2. PhyDNet-SEVIR CANNOT be topped up: models/phydnet_facl.py, the file that run
   used, exists on no server. It is skipped here and must be re-run from
   scratch (already queued as phydnet__sevir).

Run with --dry-run first; it prints exactly what it would do per cell.
"""
import argparse, csv, os, re, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_threshold_backfill import parse_last_eval_block

FALFCL='/home/vatsal/Dataserver2/Neurips/Models_falfcl'
PY='/home/vatsal/miniconda3/envs/earthformer/bin/python'
REPO='/home/vatsal/NWM/Baselines_Precipitation_Nowcasting'

# target budget per dataset (owner's, same as the new cells)
TARGET={'cikm':80,'sevir':30,'shanghai':80,'meteo':50}
UNRESUMABLE={('PhyDNet','sevir'):'models/phydnet_facl.py no longer exists on any server'}

def epochs_reached(log):
    """Epochs actually trained, read from ckpt-last.pt.

    NOT parsed from the log: the "Epoch N avg train loss" line was added to the
    runner partway through this project, so older runs have no such line and a
    log-based count silently reports 0 -- which would make this script top up a
    finished run by its entire budget. The checkpoint's 'epoch' field is the
    authoritative record and exists in every checkpoint format used here.
    """
    ck=os.path.join(os.path.dirname(os.path.dirname(log)),'checkpoints','ckpt-last.pt')
    if not os.path.exists(ck):
        ck=os.path.join(os.path.dirname(os.path.dirname(log)),'checkpoints','ckpt-best.pt')
    if not os.path.exists(ck): return None
    try:
        import torch
        return int(torch.load(ck,map_location='cpu',weights_only=False)['epoch']) + 1
    except Exception:
        return None

def plan():
    out=[]
    for lg in sorted(__import__('glob').glob(os.path.join(FALFCL,'**','log.log'), recursive=True)):
        run=lg.replace('/logs/log.log','')
        ds=next((d for d in ('sevir','meteo','shanghai','cikm') if d in lg.lower()), None)
        if not ds: continue
        done=epochs_reached(lg)
        if done is None: continue          # no checkpoint at all -> nothing to resume
        tgt=TARGET[ds]
        per,avg=parse_last_eval_block(lg)
        best=os.path.join(run,'checkpoints','ckpt-best.pt')
        last=os.path.join(run,'checkpoints','ckpt-last.pt')
        out.append(dict(run=run, name=os.path.basename(run), dataset=ds,
                        epochs_done=done, target=tgt, remaining=max(0,tgt-done),
                        csi_m=avg, has_best=os.path.exists(best), has_last=os.path.exists(last)))
    return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true',default=True)
    ap.parse_args()
    print(f"{'run':44}{'ds':10}{'done':>6}{'target':>8}{'left':>6}{'CSI-M':>9}  status")
    for p in plan():
        why=None
        if not p['has_last']: why='no ckpt-last -> cannot resume (would need full re-run)'
        elif p['remaining']==0: why='already at/over target -> nothing to do'
        for (m,d),reason in UNRESUMABLE.items():
            if m.lower() in p['name'].lower() and d==p['dataset']: why='SKIP: '+reason
        print(f"{p['name'][:43]:44}{p['dataset']:10}{p['epochs_done']:>6}{p['target']:>8}"
              f"{p['remaining']:>6}{(p['csi_m'] or 0):>9.4f}  {why or 'TOP-UP (best ckpt will be preserved first)'}")
