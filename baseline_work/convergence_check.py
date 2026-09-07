#!/usr/bin/env python
"""Was a completed baseline still improving when training stopped?

The owner's evidence bar, explicitly: a baseline qualifies for extension only if
its VALIDATION CSI-M CURVE was still meaningfully improving over the last few
checkpoints -- NOT because a number "looks lower than expected". EarthFarseer
scoring below ConvLSTM is not evidence; a still-rising curve is.

Rule applied (stated so it is auditable, and deliberately conservative):
  a run is NON-CONVERGED iff
     (1) the best validation CSI-M occurs in the final 25% of validations, AND
     (2) best(last third) - best(first two thirds) > 0.005 CSI-M
  Otherwise it is CONVERGED (plateaued or declining) and per the owner's rule
  this is the best-checkpoint rule working, not a reduction - no action.

DAWN-Cast is excluded entirely: it keeps its original fixed budget throughout.
"""
import csv, glob, os, re, sys

VAL_RE = re.compile(r"Valid Results: ([0-9]*\.[0-9]+)")
EXCLUDE = ('dawncast', 'lastocast', 'lpcast', 'amplinet')     # DAWN-Cast + its aliases
MIN_GAIN = 0.005

def curve(log):
    return [float(x) for x in VAL_RE.findall(open(log, errors='ignore').read())]

def classify(v):
    if len(v) < 4: return 'too-short', f"only {len(v)} validations", 0.0
    n = len(v); tail_start = int(n * 0.75)
    best_i = max(range(n), key=lambda i: v[i])
    third = max(1, n // 3)
    gain = max(v[-third:]) - max(v[:-third])
    still_rising = best_i >= tail_start and gain > MIN_GAIN
    why = (f"best at validation {best_i+1}/{n}; last-third best exceeds earlier best "
           f"by {gain:+.4f}")
    return ('NON-CONVERGED' if still_rising else 'converged'), why, gain

def scan(roots):
    rows = []
    for r in roots:
        for lg in sorted(glob.glob(os.path.join(r, '**', 'log.log'), recursive=True)):
            name = lg.replace('/logs/log.log', '')
            base = os.path.basename(name).lower()
            if any(k in base for k in EXCLUDE):
                continue
            v = curve(lg)
            if not v: continue
            verdict, why, gain = classify(v)
            rows.append((os.path.basename(name), verdict, why, gain, v))
    return rows

if __name__ == '__main__':
    roots = sys.argv[1:] or [
        '/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/Exps/baselines_falfcl',
        '/home/vatsal/Dataserver2/Neurips/Models_falfcl',
    ]
    rows = scan(roots)
    rows.sort(key=lambda r: (r[1] != 'NON-CONVERGED', -r[3]))
    print(f"{'run':40}{'verdict':16}evidence")
    for name, verdict, why, gain, v in rows:
        print(f"  {name[:38]:40}{verdict:16}{why}")
        print(f"  {'':40}{'':16}curve: " + " ".join(f"{x:.3f}" for x in v[-10:]) + "  (last 10)")
    nc = [r for r in rows if r[1] == 'NON-CONVERGED']
    print(f"\n{len(nc)} of {len(rows)} runs qualify for extension: {[r[0] for r in nc] or 'none'}")
    print("DAWN-Cast excluded by rule; not evaluated.")
