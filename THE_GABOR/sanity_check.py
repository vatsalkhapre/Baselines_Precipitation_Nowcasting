"""
Pre-training sanity checks for Experiment 1.

    python -m THE_GABOR.sanity_check                 # full (touches SEVIR data)
    python -m THE_GABOR.sanity_check --skip_data     # model/logging checks only

Prints a clear pass/fail report.  Nothing here starts real training.
"""

import argparse
import hashlib
import os
import os.path as osp
import subprocess
import sys
import tempfile

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import torch

from THE_GABOR.models.gabor_mlp_model import GaborMLPControlled, get_model
from THE_GABOR.utils import gabor_logging as glog
from THE_GABOR.utils import gabor_visualization as gviz
from THE_GABOR.utils.gabor_probe import build_probe, probe_gabor_layer, select_neurons
from THE_GABOR.utils.init_checkpoint import (architecture_signature,
                                             create_initial_checkpoint,
                                             load_initial_checkpoint,
                                             sha256_file)

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((name, True, detail or ''))
        print(f'  [PASS] {name}' + (f'  --  {detail}' if detail else ''))
    except Exception as e:                                   # noqa: BLE001
        RESULTS.append((name, False, repr(e)))
        print(f'  [FAIL] {name}  --  {e!r}')


CFG = dict(model='GaborMLPControlled', space='pixel', img_channel=1, frames_in=5, frames_out=20,
           hidden_dim=64, wave='db6', wavelet_level=2, hf_mode='separate',
           freq_multiplier=1.0, weight_scale=0.1, alpha=1.0, beta=1.0,
           size_factor=1.0)


def build(cfg=CFG, total_steps=100):
    return get_model(T_in=cfg['frames_in'], T_out=cfg['frames_out'],
                     img_channels=cfg['img_channel'], dim=cfg['hidden_dim'],
                     weight_scale=cfg['weight_scale'], alpha=cfg['alpha'],
                     beta=cfg['beta'], freq_multiplier=cfg['freq_multiplier'],
                     size_factor=cfg['size_factor'], wave=cfg['wave'],
                     wavelet_level=cfg['wavelet_level'], hf_mode=cfg['hf_mode'],
                     total_steps=total_steps, const_ratio=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip_data', action='store_true')
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()
    device = torch.device(args.device)

    tmp = tempfile.mkdtemp(prefix='the_gabor_sanity_')
    print('=' * 70)
    print('THE_GABOR -- Experiment 1 sanity checks')
    print('=' * 70)

    state = {}

    # 1 ------------------------------------------------------------------
    def c1():
        m = build().to(device)
        state['model'] = m
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        subs = list(m.gabor_layers().keys())
        return f'params={n/1e6:.3f}M subbands={subs}'
    check('1. model imports and builds', c1)

    # 2 / 3 --------------------------------------------------------------
    def c2():
        m = state['model']
        x = torch.randn(2, 5, 1, 128, 128, device=device)
        state['x'] = x
        y = m(x)
        state['y'] = y
        return f'in={tuple(x.shape)} out={tuple(y.shape)}'
    check('2. forward pass runs', c2)

    def c3():
        y = state['y']
        assert tuple(y.shape) == (2, 20, 1, 128, 128), f'unexpected {tuple(y.shape)}'
        return str(tuple(y.shape))
    check('3. output shape matches (B, T_out, C, H, W)', c3)

    # 4 ------------------------------------------------------------------
    def c4():
        net = state['model'].net
        x = torch.randn(2, 64, 128, 128, device=device)
        ll, hf = net.dwt(x)
        rec = net.idwt((ll, hf))[..., :128, :128]
        err = (rec - x).abs().max().item()
        assert err < 1e-3, f'DWT/IDWT round trip error {err}'
        shapes = [tuple(ll.shape)] + [tuple(h.shape) for h in hf]
        return (f'wave={net.wave} J={net.level} subband shapes={shapes} '
                f'roundtrip_max_err={err:.2e}')
    check('4. DWT / IDWT round trip for the selected wavelet+level', c4)

    # 5 / 6 / 7 / 8 ------------------------------------------------------
    def c58():
        m = state['model']
        m.zero_grad(set_to_none=True)
        x = torch.randn(1, 5, 1, 64, 64, device=device)
        gt = torch.randn(1, 20, 1, 64, 64, device=device)
        pred, loss = m.predict(frames_in=x, frames_gt=gt, compute_loss=True)
        assert set(loss.keys()) == {'facl_loss', 'total_loss'}, loss.keys()
        assert loss['total_loss'] is loss['facl_loss'], 'total_loss is not the FACL tensor'
        loss['total_loss'].backward()
        state['loss'] = loss
        state['grads'] = {n: (p.grad is not None and torch.isfinite(p.grad).all().item()
                              and p.grad.abs().sum().item() > 0)
                          for n, p in m.named_parameters()}
        return f"facl={loss['facl_loss'].item():.4f}"
    check('5-8. FACL-only loss, backward runs', c58)

    def c5():
        g = state['grads']
        gabor = {k: v for k, v in g.items() if '.gabor.' in k}
        assert gabor, 'no gabor parameters found'
        bad = [k for k, v in gabor.items() if not v]
        assert not bad, f'no/zero gradient for {bad}'
        return f'{len(gabor)} Gabor tensors received non-zero gradients'
    check('5. Gabor receives gradients', c5)

    def c6():
        g = state['grads']
        mlp = {k: v for k, v in g.items() if '.mlp.' in k}
        assert mlp, 'no mlp parameters found'
        bad = [k for k, v in mlp.items() if not v]
        assert not bad, f'no/zero gradient for {bad}'
        return f'{len(mlp)} MLP tensors received non-zero gradients'
    check('6. MLP receives gradients', c6)

    def c7():
        loss = state['loss']
        assert loss['total_loss'] is loss['facl_loss']
        return 'total_loss is the FACL tensor itself (no other term added)'
    check('7. only FACL contributes to total loss', c7)

    def c8():
        loss = state['loss']
        d = abs(loss['total_loss'].item() - loss['facl_loss'].item())
        assert d == 0.0, d
        return f'|total - facl| = {d}'
    check('8. total_loss == facl_loss to numerical precision', c8)

    # 9 / 10 -------------------------------------------------------------
    def c9():
        sig = architecture_signature(CFG)
        path = osp.join(tmp, f'initial_pixel_{sig}_seed0.pt')
        create_initial_checkpoint(build, path, 0, CFG)
        state['init_path'], state['sig'] = path, sig

        m_random, m_storm = build(), build()
        n_r = sum(p.numel() for p in m_random.parameters() if p.requires_grad)
        n_s = sum(p.numel() for p in m_storm.parameters() if p.requires_grad)
        assert n_r == n_s, (n_r, n_s)
        state['m_random'], state['m_storm'] = m_random, m_storm
        return f'RANDOM params={n_r} == STORM params={n_s}'
    check('9. RANDOM and STORM models have identical parameter counts', c9)

    def c10():
        path, sig = state['init_path'], state['sig']
        # what each run copies into its own run directory
        p_random = osp.join(tmp, 'run_random_initial_model.pt')
        p_storm = osp.join(tmp, 'run_storm_initial_model.pt')
        import shutil
        shutil.copyfile(path, p_random)
        shutil.copyfile(path, p_storm)
        h_r, h_s = sha256_file(p_random), sha256_file(p_storm)
        assert h_r == h_s, 'initial checkpoints differ byte-for-byte'

        load_initial_checkpoint(state['m_random'], path, sig)
        load_initial_checkpoint(state['m_storm'], path, sig)
        sd_r, sd_s = state['m_random'].state_dict(), state['m_storm'].state_dict()
        assert sd_r.keys() == sd_s.keys()
        diffs = [k for k in sd_r if not torch.equal(sd_r[k], sd_s[k])]
        assert not diffs, f'tensors differ after loading: {diffs[:5]}'
        return f'sha256={h_r[:16]}...  all {len(sd_r)} tensors identical'
    check('10. RANDOM and STORM initial checkpoints byte-for-byte identical', c10)

    # 11 -----------------------------------------------------------------
    def c11():
        scal = glog.gabor_parameter_scalars(state['model'])
        need = ['gabor/LL/freq/mean', 'gabor/LL/freq/std', 'gabor/LL/freq/min',
                'gabor/LL/freq/max', 'gabor/LL/effective_frequency/mean',
                'gabor/LL/gamma/mean', 'gabor/LL/mu/mean',
                'gabor/LL/linear_weight/mean', 'gabor/LL/linear_bias/mean',
                'gabor/HF_level_1/freq/mean', 'gabor/HF_level_2/freq/mean']
        missing = [k for k in need if k not in scal]
        assert not missing, f'missing keys {missing}'
        assert all(np.isfinite(v) for v in scal.values())
        return f'{len(scal)} scalar keys, e.g. LL freq mean={scal["gabor/LL/freq/mean"]:.4f}'
    check('11. Gabor parameter logging works', c11)

    # 12 / 13 / 14 -------------------------------------------------------
    def c12():
        s, xp = build_probe(5)
        state['probe'] = (s, xp)
        layer = state['model'].gabor_layers()['LL']
        r = probe_gabor_layer(layer, xp)
        sin = r['sinusoid']
        assert sin.shape == (len(s), layer.linear.out_features)
        assert np.isfinite(sin).all() and np.abs(sin).max() <= 1.0 + 1e-6
        return f'sin(z) shape={sin.shape} range=[{sin.min():.3f}, {sin.max():.3f}]'
    check('12. raw sinusoid probe works', c12)

    def c13():
        s, xp = state['probe']
        layer = state['model'].gabor_layers()['LL']
        r = probe_gabor_layer(layer, xp)
        g = r['gabor']
        direct = layer(xp.to(layer.linear.weight.device)).detach().cpu().numpy()
        err = np.abs(g - direct).max()
        assert err < 1e-5, f'probe disagrees with layer forward by {err}'
        return (f'Gabor(x) shape={g.shape} range=[{g.min():.3f}, {g.max():.3f}] '
                f'max|probe-forward|={err:.2e}')
    check('13. complete Gabor response probe works and matches layer forward', c13)

    def c14():
        s1, x1 = build_probe(5)
        s2, x2 = build_probe(5)
        assert torch.equal(s1, s2) and torch.equal(x1, x2)
        h = hashlib.sha256(x1.numpy().tobytes()).hexdigest()[:16]
        n1 = select_neurons(20, 4)
        n2 = select_neurons(20, 4)
        assert n1 == n2
        return f'probe sha256={h} neurons={n1} (deterministic, no RNG)'
    check('14. the same deterministic probe/neurons are used across runs', c14)

    # 15 -----------------------------------------------------------------
    def c15():
        import wandb
        os.environ.setdefault('WANDB_SILENT', 'true')
        run = wandb.init(project='THE_GABOR', name='sanity_check', mode='disabled',
                         config={'model': 'GaborMLPControlled', 'FACL_only': True},
                         dir=tmp)
        curves = glog.gabor_probe_curves(state['model'], *state['probe'], 4)
        figs = gviz.make_gabor_figures(curves, 'sanity', save_dir=osp.join(tmp, 'plots'))
        figs.update(gviz.make_mean_figures(curves, 'sanity', save_dir=osp.join(tmp, 'plots')))
        payload = glog.gabor_parameter_scalars(state['model'])
        payload.update(glog.mean_curve_summaries(curves))
        payload.update({k: wandb.Image(v) for k, v in figs.items()})
        payload.update(glog.gabor_histograms(state['model'], wandb))
        payload.update({'loss/facl': 1.0, 'loss/total': 1.0})
        wandb.log(payload, step=0)
        gviz.close_figures(figs)
        run.finish()
        pngs = sum(len(f) for _, _, f in os.walk(osp.join(tmp, 'plots')))
        return f'{len(payload)} keys logged, {pngs} local plots written'
    check('15. W&B logging + local plots work', c15)

    # 16 -----------------------------------------------------------------
    def c16():
        m = state['model']
        names = []
        for name in ('initial_model.pt', 'best_model.pt', 'final_model.pt'):
            p = osp.join(tmp, name)
            torch.save({'model': m.state_dict()}, p)
            names.append(name)
        gp = osp.join(tmp, 'gabor_state.pt')
        torch.save({'gabor': glog.gabor_state_dict(m)}, gp)
        gs = torch.load(gp, map_location='cpu', weights_only=False)['gabor']
        for sub, d in gs.items():
            for k in ('freq', 'effective_frequency', 'gamma', 'mu',
                      'linear.weight', 'linear.bias'):
                assert k in d, f'{sub} missing {k}'
        return f'saved {names} + gabor_state.pt (subbands={list(gs.keys())})'
    check('16. checkpoints and gabor_state save/load correctly', c16)

    # 17 / 18 ------------------------------------------------------------
    if args.skip_data:
        print('  [SKIP] 17/18 pixel SEVIR regime filtering (--skip_data)')
    else:
        from THE_GABOR.datasets.sevir_regime_dataset import (
            build_sevir_regime_dataset, dataset_stats, regime_sanity_report)

        def make_dc(regime):
            def _c():
                ds = build_sevir_regime_dataset('val', regime, img_size=128,
                                                seq_len=25, stride=13, batch_size=4)
                ok, msg = regime_sanity_report(ds, regime)
                assert ok, msg
                st = dataset_stats(ds)
                sample = ds[0]
                assert tuple(sample.shape)[1:] == (25, 1, 128, 128), tuple(sample.shape)
                return (f'{msg} | events={st["num_events"]} '
                        f'sequences={st["num_sequences"]} batch={tuple(sample.shape)}')
            return _c
        check('17. pixel SEVIR filtering returns ONLY RANDOM events', make_dc('random'))
        check('18. pixel SEVIR filtering returns ONLY STORM events', make_dc('storm'))

    # 19 -----------------------------------------------------------------
    def c19():
        repo = osp.dirname(osp.dirname(osp.abspath(__file__)))
        baseline_path = osp.join(repo, 'THE_GABOR', 'configs', 'repo_baseline.txt')
        lines = [l.rstrip('\n') for l in open(baseline_path)]
        baseline_status = {l for l in lines
                           if l and not l.startswith('#') and not l.startswith('SHA256 ')}
        baseline_sha = {}
        for l in lines:
            if l.startswith('SHA256 '):
                _, h, f = l.split(' ', 2)
                baseline_sha[f] = h

        out = subprocess.run(['git', 'status', '--porcelain'], cwd=repo,
                             capture_output=True, text=True).stdout
        current = {l for l in out.splitlines()
                   if l and not l[3:].strip().strip('"').startswith('THE_GABOR/')}
        new_changes = sorted(current - baseline_status)
        assert not new_changes, ('working tree changed outside THE_GABOR/: '
                                 + '; '.join(new_changes))

        bad = []
        for f, h in baseline_sha.items():
            cur = hashlib.sha256(open(osp.join(repo, f), 'rb').read()).hexdigest()
            if cur != h:
                bad.append(f)
        assert not bad, f'protected files changed: {bad}'
        return (f'git status outside THE_GABOR/ unchanged vs baseline; '
                f'{len(baseline_sha)} protected files byte-identical')
    check('19. no files outside THE_GABOR/ modified', c19)

    # -------------------------------------------------------------------
    print('=' * 70)
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    n_fail = len(RESULTS) - n_pass
    print(f'SANITY CHECK REPORT: {n_pass} passed, {n_fail} failed')
    for name, ok, detail in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}"
              + (f'  --  {detail}' if detail else ''))
    print('=' * 70)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
