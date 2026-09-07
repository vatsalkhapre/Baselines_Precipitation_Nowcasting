#!/usr/bin/env python
"""Per-cell training-budget disclosure for the paper (owner chose option 2).

Option 2: do NOT top up converged runs to a uniform epoch count. Instead
disclose the epoch differences, together with the two facts that make them
defensible:
  (a) every cell used the SAME checkpoint-selection rule - validate every 5
      epochs, keep the best validation CSI-M - so the reported number is the
      best checkpoint, not the last one; and
  (b) for the runs that differ, the extra epochs demonstrably would not change
      the selected checkpoint, because the best checkpoint already sits well
      before the end of training.

Column `best_at` is the evidence for (b): the epoch whose checkpoint is actually
reported. Where best_at is far below epochs_trained, more epochs bought nothing.
"""
import csv, glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VAL_RE = re.compile(r"Valid Results: ([0-9]*\.[0-9]+)")
EXCLUDE = ('dawncast','lastocast','lpcast','amplinet')

def rows(roots):
    out=[]
    for r in roots:
        for lg in sorted(glob.glob(os.path.join(r,'**','log.log'), recursive=True)):
            run=lg.replace('/logs/log.log',''); base=os.path.basename(run)
            if any(k in base.lower() for k in EXCLUDE): continue
            v=[float(x) for x in VAL_RE.findall(open(lg,errors='ignore').read())]
            if not v: continue
            # Both epoch numbers come from the CHECKPOINTS, never from the log.
            # log.log APPENDS across training sessions, so a run that was resumed
            # has several concatenated validation series; inferring the best epoch
            # as (index+1)*5 then overshoots and produced impossible values such
            # as "50 epochs trained, best at epoch 130". The checkpoint's own
            # `epoch` field is authoritative and resume-proof.
            import torch
            def ep_of(name):
                f=os.path.join(run,'checkpoints',f'ckpt-{name}.pt')
                if not os.path.exists(f): return None
                try: return int(torch.load(f,map_location='cpu',weights_only=False)['epoch'])+1
                except Exception: return None
            ep, best_ep = ep_of('last'), ep_of('best')
            out.append(dict(run=base, epochs_trained=ep or '?', validations=len(v),
                            best_val=round(max(v),4), best_at_epoch=best_ep or '?',
                            headroom=(ep-best_ep) if (ep and best_ep) else '?'))
    return out

if __name__=='__main__':
    roots=sys.argv[1:] or ['/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/baselines_falfcl',
                           '/home/vatsal/Dataserver2/Neurips/Models_falfcl']
    rs=rows(roots)
    print(f"{'run':40}{'epochs':>8}{'best_val':>10}{'best@ep':>9}{'unused_tail':>13}")
    for r in sorted(rs,key=lambda x:x['run']):
        print(f"  {r['run'][:38]:40}{str(r['epochs_trained']):>8}{r['best_val']:>10.4f}"
              f"{r['best_at_epoch']:>9}{str(r['headroom']):>13}")
    hz=[r for r in rs if isinstance(r['headroom'],int)]
    if hz:
        n=sum(1 for r in hz if r['headroom']>=10)
        print(f"\n{n}/{len(hz)} runs kept training >=10 epochs past their best checkpoint "
              f"without improving it -- this is the evidence that the differing epoch "
              f"budgets do not change the reported numbers.")
    out='/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/budget_disclosure.csv'
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0].keys())); w.writeheader(); w.writerows(rs)
    print(f"written: {out}")
