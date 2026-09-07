"""
Per-threshold CSI on top of the repository's test Evaluator.

`utils/metrics.py::Evaluator` computes CSI for every threshold internally but
only returns the threshold-AVERAGE (`csi`). `results_table.tex` needs the two
highest intensity thresholds per dataset (SEVIR: CSI-181 / CSI-219; MeteoNet:
CSI-24 / CSI-32; Shanghai and CIKM: CSI-35 / CSI-40), so those values have to be
recovered.

This subclasses the existing Evaluator and re-derives per-threshold CSI from the
hit/miss/false-alarm counters it already accumulates, using the identical
formula (`utils/metrics.py:317`):

    csi_t = mean_b(hits) / (mean_b(hits) + mean_b(misses) + mean_b(falsealarms))

averaged over lead time, exactly as `avg_csi` is built. `utils/metrics.py` is
NOT modified; `done()` is called through `super()` so the averaged metrics stay
bit-identical to a normal evaluation.

Adds to the returned dict:
    csi_t<threshold>       e.g. csi_t181, csi_t219
    csi_high, csi_high2    the two highest thresholds (csi_high = highest)
"""

import numpy as np

from utils.metrics import Evaluator as _BaseEvaluator


class PerThresholdEvaluator(_BaseEvaluator):
    """Drop-in replacement for utils.metrics.Evaluator that also emits CSI per threshold."""

    def per_threshold_csi(self):
        out = {}
        for threshold in self.thresholds:
            m = self.metrics[threshold]
            hits = np.nan_to_num(np.array(m['hits']))
            misses = np.nan_to_num(np.array(m['misses']))
            fas = np.nan_to_num(np.array(m['falsealarms']))
            if hits.size == 0:
                out[int(threshold)] = float('nan')
                continue
            denom = (hits.mean(axis=0) + misses.mean(axis=0) + fas.mean(axis=0))
            csi_t = np.nan_to_num(hits.mean(axis=0) / denom)
            out[int(threshold)] = float(np.mean(csi_t))
        return out

    def done(self, *a, **kw):
        res = super().done(*a, **kw)
        per = self.per_threshold_csi()
        for t, v in per.items():
            res[f'csi_t{t}'] = v
        ordered = sorted(per)                       # ascending threshold
        if len(ordered) >= 1:
            res['csi_high'] = per[ordered[-1]]      # highest intensity
            res['csi_high_threshold'] = ordered[-1]
        if len(ordered) >= 2:
            res['csi_high2'] = per[ordered[-2]]     # second highest
            res['csi_high2_threshold'] = ordered[-2]
        return res
