#!/usr/bin/env python
"""Unblocks the trajgru/earthformer cells once their resume gate finishes.

Runs unattended on .66. Waits for /tmp/gates_done, checks BOTH resume logs for
the two things the gate must show -- the epoch counter continuing, and the
best-checkpoint state being restored -- then flips those cells from 'blocked' to
'pending' and releases the temporary GPU reservation held for the gate.

If a gate did not pass, the cells stay blocked and the reservation stays, so the
owner sees exactly why in the morning rather than finding ungated cells running.
"""
import csv, fcntl, os, re, time, datetime

MAN='/home/vatsal/Dataserver2/Neurips/baseline_manifest/manifest.csv'
LOCK='/home/vatsal/Dataserver2/Neurips/baseline_manifest/manifest.lock'
RES='/home/vatsal/Dataserver2/Neurips/baseline_manifest/reserved_gpus.txt'
LOG='/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/AUDIT_LOG.md'
GATES={'traj_gru_falfcl_v2':'/tmp/resume_tg2.log','earthformer_falfcl_v2':'/tmp/resume_ef2.log'}

def note(msg):
    with open(LOG,'a') as f:
        f.write(f"\n[gate_watcher {datetime.datetime.now():%Y-%m-%d %H:%M}] {msg}\n")
    print(msg, flush=True)

def gate_passed(path):
    if not os.path.exists(path): return False, "log missing"
    t=open(path,errors='ignore').read()
    resumed = re.search(r"Current epoch (\d+)", t)
    restored = "Restored best-ckpt state" in t
    if not resumed:  return False, "no 'Current epoch' line - resume did not happen"
    if not restored: return False, "no 'Restored best-ckpt state' line"
    return True, f"resumed at epoch {resumed.group(1)}, best-ckpt state restored"

def main():
    while not os.path.exists('/tmp/gates_done'):
        time.sleep(60)
    results={}
    for bb,path in GATES.items():
        ok,why=gate_passed(path); results[bb]=(ok,why)
        note(f"GATE 2 {bb}: {'PASS' if ok else 'FAIL'} - {why}")
    passed={bb for bb,(ok,_) in results.items() if ok}
    if not passed:
        note("no gate passed; cells stay blocked and the GPU reservation stays."); return
    lock=open(LOCK,'w'); fcntl.flock(lock, fcntl.LOCK_EX)
    rows=list(csv.DictReader(open(MAN))); hdr=rows[0].keys(); n=0
    for r in rows:
        if r['status']=='blocked' and r['backbone'] in passed:
            r['status']='pending'; n+=1
            r['note']=r['note'].replace('blocked: awaiting smoke+resume gate for this backbone','gates passed')
            r['last_updated']=datetime.datetime.now().isoformat(timespec='seconds')
    tmp=MAN+'.tmp'
    with open(tmp,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(hdr)); w.writeheader(); w.writerows(rows)
    os.rename(tmp,MAN); fcntl.flock(lock,fcntl.LOCK_UN); lock.close()
    note(f"unblocked {n} cells for {sorted(passed)}")
    if len(passed)==len(GATES):
        keep=[l for l in open(RES) if 'TEMPORARY' not in l]
        open(RES,'w').writelines(keep)
        note("released the temporary .66 gpu2 reservation; dispatchers may now use it")

if __name__=='__main__':
    main()
