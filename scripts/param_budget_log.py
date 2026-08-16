"""
Parameter_budget.csv writer for the Gabor-initialized parameter-minimization study.

Two modes:

  seed-baseline : write a baseline row from an already-completed run directory
                  (parses its logs/log.log + params.yaml).

  append        : after a training+eval pair finishes, parse that run's
                  logs/log.log and append its row.

Both modes are idempotent on (dataset, config_id): an existing row with the same
key is replaced, so re-running an experiment updates its row instead of
duplicating it.

Nothing here modifies run_alphapre_convlstm.py or dawncast.py -- it only reads
the artifacts they already produce.

    python3 scripts/param_budget_log.py seed-baseline \
        --run_dir Exps/Gabor_sweep_runs/Meteonet_pixel_flow1.09_fhigh1.12 \
        --dataset meteo --config_id MET-B0 --notes "Gabor sweep best; budget reference"

    python3 scripts/param_budget_log.py append \
        --run_dir Exps/meteonet_param_reduction/MET-R2_nb16_hsf4 \
        --dataset meteo --config_id MET-R2 --baseline_params 59466369
"""

import argparse
import ast
import csv
import os
import os.path as osp
import re
import sys

import yaml

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEFAULT_CSV = osp.join(REPO, 'Parameter_budget.csv')

COLUMNS = [
    'dataset', 'config_id', 'status',
    'param_count', 'param_pct_of_baseline',
    'backbone', 'num_blocks', 'hidden_size_factor', 'level', 'wave',
    'k_spatial', 'hidden_dim', 'hf_mode', 'T_in', 'T_out', 'epochs',
    'weight_scale_low', 'alpha_low', 'beta_low', 'freq_multiplier_low',
    'weight_scale_high', 'alpha_high', 'beta_high', 'freq_multiplier_high',
    'gabor_source_run',
    'best_val_csi', 'best_val_epoch',
    'test_csi', 'test_csi4', 'test_csi16', 'test_hss',
    'test_mse', 'test_mae', 'test_rmse', 'test_psnr', 'test_ssim',
    'test_crps', 'test_lpips',
    'checkpoint_path', 'log_path', 'notes',
]

TEST_RE = re.compile(r"Test Results: (\{.*\})\s*$")
VAL_RE = re.compile(r"Valid Results: \{'csi': ([0-9.eE+-]+)")
PARAMS_RE = re.compile(r"Main Model Parameters: ([0-9.]+)M")


def parse_log(log_path):
    """Return (last_test_dict, val_csi_series, reported_params_M)."""
    test, vals, pm = None, [], None
    if not osp.exists(log_path):
        return test, vals, pm
    with open(log_path, errors='replace') as f:
        for line in f:
            m = TEST_RE.search(line)
            if m:
                try:
                    test = ast.literal_eval(m.group(1))
                except Exception:
                    pass
            m = VAL_RE.search(line)
            if m:
                vals.append(float(m.group(1)))
            m = PARAMS_RE.search(line)
            if m:
                pm = float(m.group(1))
    return test, vals, pm


def exact_param_count(p):
    """Recount params exactly from the run's own config (no estimate)."""
    sys.path.insert(0, REPO)
    if p.get('backbone') == 'DAWNCast_old':
        from models.DAWNCast.dawncast_old import get_model
    else:
        from models.DAWNCast.dawncast import get_model
    m = get_model(
        afno_blocks=p['spectral_blocks'],
        sparsity_threshold=p['sparsity_threshold'],
        afno_hidden_size_factor=p['spectral_hidden_size_factor'],
        weight_scale_low=p['weight_scale_low'], alpha_low=p['alpha_low'],
        beta_low=p['beta_low'], freq_multiplier_low=p['freq_multiplier_low'],
        weight_scale_high=p['weight_scale_high'], alpha_high=p['alpha_high'],
        beta_high=p['beta_high'], freq_multiplier_high=p['freq_multiplier_high'],
        size_factor=p['size_factor'], k_spatial=p['conv_kernel'],
        img_channels=p['img_channel'], dim=p['hidden_dim'],
        T_in=p['frames_in'], T_out=p['frames_out'],
        wave=p['wave'], wavelet_level=p['wavelet_level'], hf_mode=p['hf_mode'],
        input_shape=(p['img_size'], p['img_size']),
    )
    n = sum(q.numel() for q in m.parameters() if q.requires_grad)
    del m
    return n


def build_row(run_dir, dataset, config_id, baseline_params, notes, status,
              gabor_source, recount):
    run_dir = run_dir if osp.isabs(run_dir) else osp.join(REPO, run_dir)
    params_yaml = osp.join(run_dir, 'params.yaml')
    log_path = osp.join(run_dir, 'logs', 'log.log')
    if not osp.exists(params_yaml):
        raise SystemExit(f"ERROR: no params.yaml in {run_dir}")

    p = yaml.safe_load(open(params_yaml))
    test, vals, pm = parse_log(log_path)

    n = exact_param_count(p) if recount else (int(pm * 1e6) if pm else '')

    best_val = max(vals) if vals else ''
    best_ep = (vals.index(max(vals)) + 1) * 5 if vals else ''

    ckpt = osp.join(run_dir, 'checkpoints', 'ckpt-best.pt')

    row = {
        'dataset': dataset,
        'config_id': config_id,
        'status': status,
        'param_count': n,
        'param_pct_of_baseline': (round(100.0 * n / baseline_params, 2)
                                  if baseline_params and isinstance(n, int) else ''),
        'backbone': p.get('backbone', ''),
        'num_blocks': p.get('spectral_blocks', ''),
        'hidden_size_factor': p.get('spectral_hidden_size_factor', ''),
        'level': p.get('wavelet_level', ''),
        'wave': p.get('wave', ''),
        'k_spatial': p.get('conv_kernel', ''),
        'hidden_dim': p.get('hidden_dim', ''),
        'hf_mode': p.get('hf_mode', ''),
        'T_in': p.get('frames_in', ''),
        'T_out': p.get('frames_out', ''),
        'epochs': p.get('epochs', ''),
        'gabor_source_run': gabor_source,
        'best_val_csi': best_val,
        'best_val_epoch': best_ep,
        'checkpoint_path': ckpt if osp.exists(ckpt) else '(not written yet)',
        'log_path': log_path,
        'notes': notes,
    }
    for k in ('weight_scale_low', 'alpha_low', 'beta_low', 'freq_multiplier_low',
              'weight_scale_high', 'alpha_high', 'beta_high', 'freq_multiplier_high'):
        row[k] = p.get(k, '')
    for k in ('csi', 'csi4', 'csi16', 'hss', 'mse', 'mae', 'rmse',
              'psnr', 'ssim', 'crps', 'lpips'):
        row[f'test_{k}'] = test.get(k, '') if test else ''
    return row


def upsert(csv_path, row):
    rows, seen = [], False
    if osp.exists(csv_path):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                if (r.get('dataset'), r.get('config_id')) == (row['dataset'], row['config_id']):
                    rows.append(row)
                    seen = True
                else:
                    rows.append(r)
    if not seen:
        rows.append(row)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows), ('updated' if seen else 'appended')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['seed-baseline', 'append'])
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--config_id', required=True)
    ap.add_argument('--baseline_params', type=int, default=0)
    ap.add_argument('--gabor_source', default='')
    ap.add_argument('--notes', default='')
    ap.add_argument('--status', default='')
    ap.add_argument('--csv', default=DEFAULT_CSV)
    ap.add_argument('--no_recount', action='store_true',
                    help='trust the logged "Main Model Parameters" instead of rebuilding')
    a = ap.parse_args()

    status = a.status or ('baseline' if a.mode == 'seed-baseline' else 'complete')
    base = a.baseline_params
    if a.mode == 'seed-baseline' and not base:
        base = 0  # filled in below from its own count

    row = build_row(a.run_dir, a.dataset, a.config_id, base, a.notes, status,
                    a.gabor_source, recount=not a.no_recount)
    if a.mode == 'seed-baseline' and isinstance(row['param_count'], int):
        row['param_pct_of_baseline'] = 100.0

    n, what = upsert(a.csv, row)
    print(f"[{what}] {a.dataset}/{a.config_id}  params={row['param_count']:,}  "
          f"best_val_csi={row['best_val_csi']}  test_csi={row['test_csi']}  "
          f"-> {a.csv} ({n} rows)")


if __name__ == '__main__':
    main()
