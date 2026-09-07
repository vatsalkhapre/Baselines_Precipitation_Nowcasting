#!/usr/bin/env python
"""Consolidated FALFCL results table, built from log.log rather than the CSV.

models_falfcl.csv is incomplete: several completed runs (SimVP-MeteoNet,
SimVP-SEVIR, TrajGRU-SEVIR, PhyDNet-SEVIR) have checkpoints and full evaluations
but no CSV row at all, because the row is only written when a run is invoked
with --eval. Their numbers exist only in log.log. Building the table from logs
therefore covers strictly more cells than the CSV does.

Emits one row per (model, dataset) for the FALFCL protocol series, with CSI-M,
CSI-H, CSI-E (top two thresholds for that dataset), the pooled CSIs, and the
training budget actually used - so the budget inconsistencies in the existing
table are visible in the table itself rather than buried in a log.
"""
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from csi_threshold_backfill import parse_last_eval_block, THRESHOLDS, dataset_of

FALFCL_ROOT='/home/vatsal/Dataserver2/Neurips/Models_falfcl'
NEW_ROOTS=['/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/baselines_falfcl']

RES_RE=re.compile(r"Test Results: (\{.*?\})", re.S)
BS_RE=re.compile(r"Instantaneous batch size per GPU = (\d+)")
EP_RE=re.compile(r"Num Epochs = (\d+)")
PAR_RE=re.compile(r"Main Model Parameters: ([0-9.]+)M")

def model_of(path):
    b=os.path.basename(path).lower()
    for k,v in [('convlstm','ConvLSTM'),('earthfarseer','EarthFarseer'),('alphapre','AlphaPre'),
                ('phydnet','PhyDNet'),('trajgru','TrajGRU'),('traj_gru','TrajGRU'),
                ('earthformer','EarthFormer'),('mau','MAU'),('simvp','SimVP'),
                ('exprecast','exPreCast'),('wadepre','WADEPre'),('diffcast','DiffCast')]:
        if k in b: return v
    return None

def scan(roots):
    out={}
    for r in roots:
        for lg in glob.glob(os.path.join(r,'**','log.log'), recursive=True):
            run=lg.replace('/logs/log.log','')
            m,ds=model_of(run), dataset_of(lg)
            if not m or not ds: continue
            per,avg=parse_last_eval_block(lg)
            if per is None: continue
            txt=open(lg,errors='ignore').read()
            res=RES_RE.findall(txt)
            d={}
            if res:
                try: d=eval(res[-1])
                except Exception: d={}
            th=THRESHOLDS[ds]
            out[(m,ds)]=dict(model=m,dataset=ds,run=os.path.basename(run),
                csi_m=round(avg,6),
                csi_h_thr=th[-2], csi_h=round(per[th[-2]],6),
                csi_e_thr=th[-1], csi_e=round(per[th[-1]],6),
                csi_pool4x4=round(d.get('csi_pool4x4',float('nan')),6) if d else '',
                csi_pool16x16=round(d.get('csi_pool16x16',float('nan')),6) if d else '',
                hss=round(d.get('hss',float('nan')),6) if d else '',
                ssim=round(d.get('ssim',float('nan')),6) if d else '',
                mse=round(d.get('mse',float('nan')),4) if d else '',
                batch_size=(BS_RE.findall(txt) or [''])[-1],
                epochs_reached=(EP_RE.findall(txt) or [''])[-1],
                params_M=(PAR_RE.findall(txt) or [''])[-1])
    return out

if __name__=='__main__':
    rows=scan([FALFCL_ROOT]+NEW_ROOTS)
    DS=['cikm','sevir','shanghai','meteo']
    MODELS=['ConvLSTM','TrajGRU','PhyDNet','MAU','SimVP','EarthFormer','AlphaPre','EarthFarseer']
    print(f"{'model':14}" + "".join(f"{d:>26}" for d in DS))
    print(f"{'':14}" + "".join(f"{'CSI-M / CSI-H / CSI-E':>26}" for d in DS))
    print("-"*(14+26*len(DS)))
    for m in MODELS:
        line=f"{m:14}"
        for d in DS:
            r=rows.get((m,d))
            line += f"{('%.4f/%.4f/%.4f'%(r['csi_m'],r['csi_h'],r['csi_e'])):>26}" if r else f"{'-':>26}"
        print(line)
    print("\nbudget actually used (bs / epochs):")
    for m in MODELS:
        line=f"  {m:14}"
        for d in DS:
            r=rows.get((m,d))
            line += f"{(r['batch_size']+'/'+r['epochs_reached']):>12}" if r else f"{'-':>12}"
        print(line)
    out='/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/results_table_falfcl.csv'
    if rows:
        with open(out,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(next(iter(rows.values())).keys()))
            w.writeheader()
            for k in sorted(rows): w.writerow(rows[k])
        print(f"\nwritten: {out}  ({len(rows)} cells)")
