#!/usr/bin/env python
"""Per-host dispatcher for the baseline run queue.

One instance runs on each host. All instances coordinate through a single
manifest on the shared mount (/home/vatsal/Dataserver2), claimed under an
O_EXCL lock and rewritten with tmp+os.rename, so two hosts can never claim the
same cell.

GPU policy (owner's hard rule 4): use every idle GPU, never idle a GPU while
cells are pending, never touch a GPU that is busy or listed in
reserved_gpus.txt. That file is re-read before every launch, so the owner can
reclaim a card at any moment by adding a line; running jobs are not killed.

Mandatory gates that are NOT optional and are NOT skipped to save time:
  * sanity gate at ~10% of the cell's step budget (NaN / divergence / trivial CSI)
  * a cell reaching 'done' still requires the separate final verification pass
    before its number is treated as real (verify.py)
Failure marks the cell 'failed' and moves on; it does not silently burn budget.
"""
import argparse, csv, os, re, socket, subprocess, sys, time, fcntl, shutil, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = '/home/vatsal/NWM/Baselines_Precipitation_Nowcasting'
MAN_DIR = '/home/vatsal/Dataserver2/Neurips/baseline_manifest'
MANIFEST = os.path.join(MAN_DIR, 'manifest.csv')
LOCK = os.path.join(MAN_DIR, 'manifest.lock')
RESERVED = os.path.join(MAN_DIR, 'reserved_gpus.txt')
PY = '/home/vatsal/miniconda3/envs/earthformer/bin/python'
RESULTS_CSV = '/home/vatsal/Dataserver2/Neurips/csv_files/models_falfcl.csv'
WANDB_PROJECT = 'ICLR26_FACL_runs'
MIN_FREE_MIB = 24000   # a cell needs real headroom; AlphaPre peaks ~21 GiB

def host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.24.52.210', 1))
        ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())

# ---------------- manifest, locked ----------------
class Locked:
    def __enter__(self):
        self.f = open(LOCK, 'w'); fcntl.flock(self.f, fcntl.LOCK_EX); return self
    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN); self.f.close()

def read_manifest():
    with open(MANIFEST) as f:
        r = list(csv.DictReader(f)); return r, r[0].keys() if r else []

def write_manifest(rows, hdr):
    tmp = MANIFEST + f'.tmp.{os.getpid()}'
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hdr)); w.writeheader(); w.writerows(rows)
    os.rename(tmp, MANIFEST)          # atomic within the mount

def update_cell(cell_id, **kv):
    with Locked():
        rows, hdr = read_manifest()
        for r in rows:
            if r['cell_id'] == cell_id:
                r.update({k: str(v) for k, v in kv.items()})
                r['last_updated'] = datetime.datetime.now().isoformat(timespec='seconds')
        write_manifest(rows, hdr)

def claim_cell():
    """Atomically take the first pending cell. Manifest order == protocol order
    (CIKM smoke -> SEVIR headline -> Shanghai -> MeteoNet)."""
    with Locked():
        rows, hdr = read_manifest()
        for r in rows:
            if r['status'] == 'pending':
                r['status'] = 'running'; r['host'] = host_ip()
                r['attempts'] = str(int(r.get('attempts') or 0) + 1)
                r['last_updated'] = datetime.datetime.now().isoformat(timespec='seconds')
                write_manifest(rows, hdr)
                return dict(r)
    return None

# ---------------- GPUs ----------------
def reserved_here():
    out = set(); me = host_ip()
    if not os.path.exists(RESERVED): return out
    for line in open(RESERVED):
        line = line.split('#')[0].strip()
        if not line: continue
        p = line.split()
        if len(p) >= 2 and p[0] == me:
            try: out.add(int(p[1]))
            except ValueError: pass
    return out

def busy_uuids():
    """UUIDs of GPUs with ANY compute process on them.

    Free memory alone is not a safe test: a job that has just started has not
    allocated yet, so a memory-only check races two jobs onto one card. This is
    not hypothetical -- it happened once here, putting a queued cell and a
    running smoke test on the same GPU.
    """
    q = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid',
                        '--format=csv,noheader'], capture_output=True, text=True)
    return {l.strip() for l in q.stdout.strip().splitlines() if l.strip()}

def free_gpus():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True)
    res, blocked, busy = [], reserved_here(), busy_uuids()
    for line in q.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(',')]
        i, uuid, tot, used = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        if i in blocked: continue
        if uuid in busy: continue           # someone is already computing here
        if (tot - used) >= MIN_FREE_MIB: res.append(i)
    return res

# ---------------- sanity gate ----------------
LOSS_RE = re.compile(r"'total_loss': ([0-9.eE+-]+|nan|inf)")
CSI_RE  = re.compile(r"Valid Results: ([0-9.eE+-]+|None)")
STEP_RE = re.compile(r"Step (\d+)/(\d+)")
MIN_SAMPLES = 12          # below this the head/tail comparison is meaningless

def sanity_check(logfile, total_steps=None):
    """Returns (verdict, reason) where verdict is 'pass' | 'fail' | 'defer'.

    'defer' matters: the runner only logs a loss line every 200 steps, so a slow
    model can have fewer than a handful of samples after 15 minutes. Comparing a
    20-sample head against a 20-sample tail then compares a slice with ITSELF and
    reports a confident pass on no evidence (observed: "loss 86->86"). Rather
    than pass vacuously, defer and check again later.
    """
    if not os.path.exists(logfile): return 'defer', "log not yet written"
    txt = open(logfile, errors='ignore').read()
    raw = LOSS_RE.findall(txt)
    for v in raw:
        if v in ('nan', 'inf', '-inf'):
            return 'fail', f"non-finite training loss ({v})"
    vals = [float(v) for v in raw]
    if any(v != v for v in vals): return 'fail', "NaN training loss"

    csis = [c for c in CSI_RE.findall(txt) if c != 'None']
    if csis:
        best = max(float(c) for c in csis)
        if best < 0.01:
            return 'fail', f"CSI still trivially low ({best:.4g}) after a full validation"

    if len(vals) < MIN_SAMPLES:
        return 'defer', f"only {len(vals)} loss samples so far (need {MIN_SAMPLES})"

    k = max(5, len(vals) // 4)
    head = sum(vals[:k]) / k
    tail = sum(vals[-k:]) / k
    if tail > head * 3 and tail > 1e-3:
        return 'fail', f"loss diverging: first-{k} mean {head:.4g} -> last-{k} mean {tail:.4g}"
    if csis:
        return 'pass', f"loss {head:.4g}->{tail:.4g} over {len(vals)} samples, best CSI {max(float(c) for c in csis):.4f}"
    return 'pass', f"loss {head:.4g}->{tail:.4g} over {len(vals)} samples"

# ---------------- launch ----------------
def build_cmd(cell, gpu, resume_ckpt=None):
    exp_dir, exp_note = 'baselines_falfcl', f"{cell['model']}_on_{cell['dataset']}"
    cmd = [PY, os.path.join(REPO, 'run_baselines.py'),
           '--exp_dir', exp_dir, '--exp_note', exp_note,
           '--backbone', cell['backbone'], '--dataset', cell['dataset'],
           '--batch_size', cell['batch_size'], '--seq_len', '25',
           '--frames_in', cell['frames_in'], '--frames_out', cell['frames_out'],
           '--img_size', cell['img_size'], '--epochs', cell['epochs'],
           '--lr', cell['lr'], '--seed', cell['seed'], '--valid',
           '--num_workers', '8', '--wandb_state', 'online',
           '--wandb_project_name', WANDB_PROJECT,
           '--run_name', f"{cell['cell_id']}",
           '--results_csv', RESULTS_CSV]
    if resume_ckpt:
        # --res_opt is mandatory on resume: without it the optimizer, LR
        # schedule and step counter silently restart from zero.
        cmd += ['--ckpt_milestone', resume_ckpt, '--res_opt']
    return cmd

def run_cell(cell, gpu):
    exp = os.path.join(REPO, 'Exps', 'baselines_falfcl', f"{cell['model']}_on_{cell['dataset']}")
    log = os.path.join(exp, 'logs', 'log.log')
    ckpt_last = os.path.join(exp, 'checkpoints', 'ckpt-last.pt')
    resume = ckpt_last if os.path.exists(ckpt_last) else None
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))   # set BEFORE the process starts
    cmd = build_cmd(cell, gpu, resume)
    print(f"[{time.strftime('%H:%M:%S')}] LAUNCH {cell['cell_id']} on gpu{gpu}"
          + (f" (resuming {os.path.basename(resume)})" if resume else " (fresh)"), flush=True)
    proc = subprocess.Popen(cmd, env=env, cwd=REPO,
                            stdout=open(os.path.join(MAN_DIR, f"{cell['cell_id']}.stdout"), 'a'),
                            stderr=subprocess.STDOUT)
    update_cell(cell['cell_id'], gpu_index=gpu, pid=proc.pid, ckpt_path=ckpt_last,
                status='running', note=cell.get('note', ''))

    gate_done, t0 = False, time.time()
    while proc.poll() is None:
        time.sleep(60)
        # sanity gate at ~10% of budget, approximated by elapsed-vs-expected;
        # fires once, at the earliest point where enough loss history exists.
        # Mandatory sanity gate. Re-checked until it can actually reach a verdict
        # (see sanity_check docstring); it never passes on insufficient evidence.
        if not gate_done and time.time() - t0 > 900:
            verdict, why = sanity_check(log)
            if verdict == 'defer':
                if time.time() - t0 > 6 * 3600:
                    gate_done = True
                    print(f"[{time.strftime('%H:%M:%S')}] SANITY {cell['cell_id']}: "
                          f"UNRESOLVED after 6h - {why}; letting it run, flag for review", flush=True)
                continue
            gate_done = True
            print(f"[{time.strftime('%H:%M:%S')}] SANITY {cell['cell_id']}: "
                  f"{verdict.upper()} - {why}", flush=True)
            if verdict == 'fail':
                proc.kill(); proc.wait()
                update_cell(cell['cell_id'], status='failed', note=f"sanity gate: {why}")
                return 'failed'
    rc = proc.returncode
    if rc == 0:
        # MANDATORY GATE 3: verify before the cell counts as complete. Run here,
        # inline, while this dispatcher still holds the GPU -- otherwise nothing
        # runs it unattended and a bad number could sit in the table unnoticed.
        update_cell(cell['cell_id'], status='verifying', note='running final verification')
        print(f"[{time.strftime('%H:%M:%S')}] VERIFY {cell['cell_id']} on gpu{gpu}", flush=True)
        try:
            import verify as _v
            ok, msg = _v.verify(cell, gpu)
        except Exception as e:
            ok, msg = False, f"verification crashed: {type(e).__name__}: {e}"
        update_cell(cell['cell_id'], status='done' if ok else 'failed',
                    note=('verified: ' if ok else 'verification FAILED: ') + msg)
        print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if ok else 'FAILED'} "
              f"{cell['cell_id']}: {msg}", flush=True)
        return 'done' if ok else 'failed'
    # preemption / crash -> back to pending so it resumes from its checkpoint
    update_cell(cell['cell_id'], status='pending', gpu_index='', pid='',
                note=f"exited rc={rc}; will resume from ckpt-last")
    print(f"[{time.strftime('%H:%M:%S')}] REQUEUE {cell['cell_id']} (rc={rc})", flush=True)
    return 'requeued'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-jobs', type=int, default=99)
    ap.add_argument('--poll', type=int, default=120)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--gpu', type=int, default=None,
                    help='pin this worker to one GPU index. One worker per GPU lets a host '
                         'run as many cells as it has free cards (hard rule 4: never leave an '
                         'idle GPU while cells are pending). Cells are still assigned '
                         'dynamically - only the worker is pinned, never a cell.')
    a = ap.parse_args()

    # Singleton per host. Two dispatchers on one machine will each claim cells
    # and can put two jobs on one GPU; that happened during bring-up. The lock
    # is held for the process lifetime and released automatically on exit.
    _tag = f'{host_ip().replace(".","_")}' + (f'_gpu{a.gpu}' if a.gpu is not None else '')
    _sing = open(f'/tmp/dispatcher_{_tag}.singleton', 'w')
    try:
        fcntl.flock(_sing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another dispatcher already owns {_tag}; exiting", flush=True)
        return
    _sing.write(str(os.getpid())); _sing.flush()

    me = host_ip(); n = 0
    print(f"dispatcher on {socket.gethostname()} ({me})"
          + (f" pinned to gpu{a.gpu}" if a.gpu is not None else "")
          + f"; reserved here: {sorted(reserved_here())}", flush=True)
    while n < a.max_jobs:
        gpus = free_gpus()
        if a.gpu is not None:
            gpus = [g for g in gpus if g == a.gpu]
        if not gpus:
            print(f"[{time.strftime('%H:%M:%S')}] no free GPU (reserved={sorted(reserved_here())}); waiting", flush=True)
            time.sleep(a.poll); continue
        cell = claim_cell()
        if cell is None:
            # Do NOT exit. A cell that later fails or is preempted returns to
            # 'pending', and top-ups add work too -- if the worker has exited
            # there is nothing left on this GPU to pick that up, so the card sits
            # idle indefinitely. Stay alive and keep polling.
            print(f"[{time.strftime('%H:%M:%S')}] queue empty; idling (will pick up requeued/new cells)", flush=True)
            time.sleep(a.poll); continue
        if a.dry_run:
            print(f"DRY-RUN would launch {cell['cell_id']} on gpu{gpus[0]}"); 
            update_cell(cell['cell_id'], status='pending'); break
        run_cell(cell, gpus[0]); n += 1

if __name__ == '__main__':
    main()
