#!/usr/bin/env python
"""Build the LaTeX results table: FACL baselines + DAWN-Cast, 4 datasets.

Follows DAWN-Cast paper Table 1, with CSI-pool4 and CSI-pool16 added as the
owner requested. Datasets are stacked vertically (one block per dataset) rather
than side by side, which keeps 8 metric columns readable on a portrait page.

Numbers are parsed from each run's log.log rather than from the results CSV,
because the CSV is missing rows for several completed runs (a row is only
written when a run is invoked with --eval). Per (model, dataset) the run in
Exps/baselines_falfcl wins over the older Models_falfcl copy, since the former
are this round's controlled cells.

Best / second / third per column are bolded / underlined / italicised, matching
the paper. Direction is respected: MSE is lower-is-better, everything else is
higher-is-better.
"""
import glob, os, re, sys

TEST_RE = re.compile(r"Test Results: (\{.*?\})", re.S)
TH = {'sevir':(181,219), 'meteo':(24,32), 'shanghai':(35,40), 'cikm':(35,40)}
CSI_BLOCK = re.compile(r"Threshold: (\d+) with melthod 1={20}\s*\n.*?<CSI> : ([0-9.eE+-]+);", re.S)

MODEL_ORDER = ['ConvLSTM','TrajGRU','PhyDNet','MAU','SimVP','EarthFormer',
               'EarthFarseer','AlphaPre','exPreCast','DiffCast','WADEPre','DAWN-Cast']

# WADEPre is trained on its OWN native loss and curriculum, not FALFCL -- that is
# the run protocol, not an oversight (its wavelet-decomposition objective is not
# separable in the way the FALFCL substitution assumes). exPreCast uses FACL
# natively. Both are marked in the table so the caption is not misleading.
NATIVE_LOSS = {'WADEPre'}
DATASETS = [('sevir','SEVIR'), ('meteo','MeteoNet'), ('shanghai','Shanghai'), ('cikm','CIKM')]

def model_of(name):
    n = name.lower()
    if 'dawncast' in n: return 'DAWN-Cast'
    if 'diffcast' in n or 'diffphydnet' in n: return 'DiffCast'   # must precede the phydnet test
    if 'exprecast' in n: return 'exPreCast'
    if 'wadepre' in n:   return 'WADEPre'
    for k, v in [('convlstm','ConvLSTM'),('earthfarseer','EarthFarseer'),('earthformer','EarthFormer'),
                 ('alphapre','AlphaPre'),('phydnet','PhyDNet'),('trajgru','TrajGRU'),
                 ('traj_gru','TrajGRU'),('mau','MAU'),('simvp','SimVP')]:
        if k in n: return v
    return None

def dataset_of(path):
    p = path.lower()
    for k in ('sevir','meteonet','meteo','shanghai','cikm'):
        if k in p: return 'meteo' if k in ('meteo','meteonet') else k
    return None

def parse(lg):
    txt = open(lg, errors='ignore').read()
    tests = TEST_RE.findall(txt)
    if not tests: return None
    try: d = eval(tests[-1])
    except Exception: return None
    # Per-threshold CSI for the LAST evaluation block.
    # Do NOT use a fixed-size tail window: a 20-frame dataset prints six
    # thresholds x four metrics x 20 per-lead-time values, which overruns any
    # fixed window and silently yields zero thresholds (CIKM's 10-frame arrays
    # fit, so the bug looked dataset-specific). Instead scan the whole file in
    # order and let later occurrences overwrite earlier ones, so what survives
    # is the final evaluation's values.
    # Bound the scan to everything BEFORE the final "Test Results:" line, then
    # keep the last occurrence of each threshold in that region. Bounding matters:
    # thresholds are also printed by VALIDATION blocks, so an unbounded scan could
    # pair validation thresholds with a test dict if a validation ran afterwards.
    region = txt[:txt.rindex('Test Results:')]
    per = {}
    for t, c in CSI_BLOCK.findall(region):
        per[int(t)] = float(c)
    return d, per

def check(m, ds, row):
    """CSI-M must equal the mean of the per-threshold CSIs from the same block.
    This is the same consistency gate used when the CSI-H/CSI-E columns were
    first backfilled; it catches a threshold set that came from a different
    evaluation than the summary dict."""
    if row.get('_all') and row.get('csi_m') is not None:
        recomputed = sum(row['_all'].values()) / len(row['_all'])
        if abs(recomputed - row['csi_m']) > 1e-4:
            print(f"% WARNING {m}/{ds}: CSI-M {row['csi_m']:.6f} != mean of per-threshold "
                  f"CSIs {recomputed:.6f} -- thresholds may come from a different block",
                  file=sys.stderr)


def collect():
    out = {}
    roots = [('/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/baselines_falfcl', 2),
             ('/home/vatsal/Dataserver2/Neurips/dawncast_eval_collected',                   2),
             ('/home/vatsal/Dataserver2/Neurips/Models_falfcl',                            1),
             ('/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Exprecast',           1),
             ('/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative/Wadepre',             1),
             ('/home/vatsal/Dataserver2/Neurips/round2_collected',                          2),
             ('/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/cikm',              1)]
    for root, prio in roots:
        for lg in sorted(glob.glob(os.path.join(root, '**', 'log.log'), recursive=True)):
            run = lg.replace('/logs/log.log', ''); base = os.path.basename(run)
            m, ds = model_of(base), dataset_of(lg)
            if not m or not ds: continue
            if root.endswith('/Exps/cikm') and m != 'WADEPre': continue
            # Owner: keep the ORIGINAL exPreCast results. The round-2 reruns
            # (drop_path_rate 0.2) did not improve them, so they are not used.
            if root.endswith('/round2_collected') and m == 'exPreCast': continue
            r = parse(lg)
            if not r: continue
            d, per = r
            hi, lo = TH[ds]
            if hi not in per or lo not in per: continue
            key = (m, ds)
            if key in out and out[key]['prio'] >= prio: continue
            out[key] = dict(prio=prio, run=base, _all=per,
                            csi_m=d.get('csi'), csi_h=per[hi], csi_e=per[lo],
                            # the TEST evaluator (utils/metrics.py) names the pooled
                            # CSIs 'csi4'/'csi16'; the VALIDATION evaluator
                            # (utils/metrics_valid.py) calls the same quantities
                            # 'csi_pool4x4'/'csi_pool16x16'. Accept either, preferring
                            # the test names, so a val-only dict never sneaks in silently.
                            p4=d.get('csi4', d.get('csi_pool4x4')),
                            p16=d.get('csi16', d.get('csi_pool16x16')),
                            hss=d.get('hss'), ssim=d.get('ssim'), mse=d.get('mse'))
    for (m, ds), row in out.items():
        check(m, ds, row)
    return out

COLS = [('csi_m','CSI-M',1),('csi_h',None,1),('csi_e',None,1),
        ('p4','CSI-pool4',1),('p16','CSI-pool16',1),
        ('hss','HSS',1),('ssim','SSIM',1),('mse','MSE',-1)]

def rank_marks(vals, direction):
    """indices of best/2nd/3rd, respecting direction (+1 higher better)."""
    idx = [i for i, v in enumerate(vals) if v is not None]
    idx.sort(key=lambda i: vals[i], reverse=(direction > 0))
    return idx[:1], idx[1:2], idx[2:3]

def fmt(v, col, b, s, t, i):
    if v is None: return '--'
    x = f"{v:.2f}" if col == 'mse' else f"{v:.4f}"
    if i in b: return r'\textbf{%s}' % x
    if i in s: return r'\underline{%s}' % x
    if i in t: return r'\textit{%s}' % x
    return x

def main():
    data = collect()
    L = []
    L.append(r'\begin{table}[t]')
    L.append(r'\centering')
    L.append(r'\caption{Performance of FACL-trained baselines and DAWN-Cast on SEVIR, MeteoNet, '
             r'Shanghai and CIKM (all $128\times128$, $T_{in}{=}5$). Best in \textbf{bold}, '
             r'second \underline{underlined}, third \textit{italic}. '
             r'CSI-H/CSI-E are the two highest intensity thresholds for each dataset. '
             r'All baselines are trained with FACL except those marked $^{\dagger}$, which '
             r'use their own native loss and curriculum as specified by their papers.}')
    L.append(r'\label{tab:facl_baselines}')
    L.append(r'\resizebox{\textwidth}{!}{%')
    L.append(r'\begin{tabular}{l|cccccccc}')
    L.append(r'\toprule')
    for ds, pretty in DATASETS:
        hi, lo = TH[ds]
        present = [m for m in MODEL_ORDER if (m, ds) in data]
        if not present: continue
        L.append(r'\multicolumn{9}{c}{\textbf{%s}} \\' % pretty)
        L.append(r'\midrule')
        L.append(r'Model & CSI-M$\uparrow$ & CSI-%d$\uparrow$ & CSI-%d$\uparrow$ & '
                 r'CSI-pool4$\uparrow$ & CSI-pool16$\uparrow$ & HSS$\uparrow$ & '
                 r'SSIM$\uparrow$ & MSE$\downarrow$ \\' % (hi, lo))
        L.append(r'\midrule')
        marks = {}
        for col, _, direction in COLS:
            marks[col] = rank_marks([data[(m, ds)][col] for m in present], direction)
        for i, m in enumerate(present):
            row = data[(m, ds)]
            cells = [fmt(row[c], c, *marks[c], i) for c, _, _ in COLS]
            name = r'\textbf{DAWN-Cast}' if m == 'DAWN-Cast' else m
            if m in NATIVE_LOSS: name += r'$^{\dagger}$'
            L.append(f'{name} & ' + ' & '.join(cells) + r' \\')
        L.append(r'\bottomrule' if ds == DATASETS[-1][0] else r'\midrule')
    L.append(r'\end{tabular}}')
    L.append(r'\end{table}')
    tex = '\n'.join(L)
    out = '/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/results_table.tex'
    open(out, 'w').write(tex + '\n')
    print(tex)
    print(f'\n% written: {out}', file=sys.stderr)
    missing = [(m, ds) for ds, _ in DATASETS for m in MODEL_ORDER if (m, ds) not in data]
    if missing:
        print('\n% MISSING CELLS: ' + ', '.join(f'{m}/{d}' for m, d in missing), file=sys.stderr)

if __name__ == '__main__':
    main()
