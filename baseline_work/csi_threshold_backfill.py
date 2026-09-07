"""Audit 5: add the two extreme-threshold CSI columns (CSI-H, CSI-E) to results.

Per the owner's clarification, "CSI-181 / CSI-219" means the top TWO CSI
thresholds for each dataset, taken from that dataset's own THRESHOLDS in the
existing evaluation code -- not lead times, and not literal 181/219 everywhere:

    SEVIR     (16,74,133,160,181,219) -> CSI-181, CSI-219
    MeteoNet  [12,18,24,32]           -> CSI-24,  CSI-32
    Shanghai  [20,30,35,40]           -> CSI-35,  CSI-40
    CIKM      [20,30,35,40]           -> CSI-35,  CSI-40

This is the same convention WADEPre's paper uses (Sec 4.1.3: "we calculate the
mean of the six thresholds (CSI-M) and select two thresholds (CSI-H and CSI-E)
for validating the extreme value benchmark").

NO TRAINING AND NO INFERENCE IS NEEDED. utils/metrics.py already evaluates every
threshold in the dataset's THRESHOLDS tuple and print_log's each one, so the
values are already sitting in every completed run's log.log. This script only
parses them.

VALIDATION GATE (required before touching any real CSV): for each run we
recompute CSI-M as the mean of the parsed per-threshold CSI values and check it
against the '[ avg_csi ]' line the evaluator itself printed. If those disagree,
the parse is wrong and the run is reported as FAILED rather than written.
"""
import re, sys, os, glob, json

THRESHOLDS = {
    'sevir':    [16, 74, 133, 160, 181, 219],
    'meteo':    [12, 18, 24, 32],
    'meteonet': [12, 18, 24, 32],
    'shanghai': [20, 30, 35, 40],
    'cikm':     [20, 30, 35, 40],
}

TH_RE  = re.compile(r"Threshold:\s*(\d+)\s*with melthod 1")
CSI_RE = re.compile(r"<CSI>\s*:\s*([0-9.eE+-]+);")
AVG_RE = re.compile(r"\[ avg_csi \]\s*:\s*([0-9.eE+-]+);")

def parse_last_eval_block(path):
    """Return (per_threshold_csi: dict, reported_avg_csi: float) for the LAST
    complete evaluation block in the log."""
    txt = open(path, errors='ignore').read()
    avgs = list(AVG_RE.finditer(txt))
    if not avgs:
        return None, None
    last = avgs[-1]
    # the block for this avg starts after the previous avg_csi line
    start = avgs[-2].end() if len(avgs) > 1 else 0
    block = txt[start:last.start()]
    ths  = TH_RE.findall(block)
    csis = CSI_RE.findall(block)
    if len(ths) != len(csis) or not ths:
        return None, None
    per = {int(t): float(c) for t, c in zip(ths, csis)}
    return per, float(last.group(1))

def dataset_of(path):
    p = path.lower()
    for k in ('sevir', 'meteonet', 'meteo', 'shanghai', 'cikm'):
        if k in p:
            return 'meteo' if k == 'meteonet' else k
    return None

def main(roots):
    rows, failures = [], []
    logs = []
    for r in roots:
        logs += glob.glob(os.path.join(r, '**', 'log.log'), recursive=True)
    for lg in sorted(logs):
        run = lg.replace('/logs/log.log', '')
        ds  = dataset_of(lg)
        per, reported = parse_last_eval_block(lg)
        name = os.path.relpath(run, os.path.dirname(os.path.dirname(run)))
        if per is None:
            failures.append((name, ds, "no parseable evaluation block"))
            continue
        if ds is None:
            failures.append((name, ds, "cannot infer dataset from path"))
            continue
        expect = THRESHOLDS[ds]
        # --- VALIDATION GATE ---
        recomputed = sum(per.values()) / len(per)
        delta = abs(recomputed - reported)
        ok_avg = delta < 1e-6
        ok_th  = sorted(per.keys()) == sorted(expect)
        if not (ok_avg and ok_th):
            failures.append((name, ds,
                f"VALIDATION FAILED: recomputed CSI-M {recomputed:.10f} vs reported {reported:.10f} "
                f"(delta {delta:.2e}); thresholds {sorted(per.keys())} vs expected {sorted(expect)}"))
            continue
        h, e = expect[-2], expect[-1]
        rows.append(dict(run=name, dataset=ds, csi_m=reported,
                         csi_h_threshold=h, csi_h=per[h],
                         csi_e_threshold=e, csi_e=per[e]))
    return rows, failures

if __name__ == '__main__':
    roots = sys.argv[1:] or [
        '/home/vatsal/Dataserver2/Neurips/Models_falfcl',
        '/home/vatsal/Dataserver2/Neurips/Baselines_Qualitative',
    ]
    rows, failures = main(roots)
    print(f"=== VALIDATED ({len(rows)}) ===")
    print(f"{'run':52}{'ds':10}{'CSI-M':>9}{'CSI-H':>16}{'CSI-E':>16}")
    for r in rows:
        print(f"{r['run'][:51]:52}{r['dataset']:10}{r['csi_m']:9.4f}"
              f"{('CSI-%d %.4f'%(r['csi_h_threshold'],r['csi_h'])):>16}"
              f"{('CSI-%d %.4f'%(r['csi_e_threshold'],r['csi_e'])):>16}")
    print(f"\n=== NOT WRITTEN ({len(failures)}) ===")
    for n, d, why in failures:
        print(f"  {n[:50]:52}{str(d):10}{why}")
    json.dump(rows, open('/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/csi_he_parsed.json','w'), indent=1)
