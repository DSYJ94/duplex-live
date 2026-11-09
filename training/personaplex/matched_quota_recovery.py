"""Owner-directed interrupted recovery on separately allocated NFS job outputs.

No native numerical admission, changed losses, fabricated success or OneDrive I/O.
"""
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from . import matched_guard as g
from .matched_job import write,visible_pair
from .matched_training_only import TrainingChildren,initial_plan,execute,HISTORY as OLD_HISTORY
from .matched_reuse import preflight,actual_usage,ORIGINAL_ROOT,DATA_SHA
from .matched_continue_policy import PACKET_SHA
from .matched_interrupted import authority,DURABLE,PRESERVED,STATE_SHA

HISTORY=dict(OLD_HISTORY,**{'8235976':dict(state='FAILED',elapsed=644,GPUs=2)})


def storage_check(root=DURABLE,home=None):
    home=Path('/home/u6ada/dsyj94.u6ada') if home is None else Path(home)
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Actual allocated NFS output directory required')
    root.resolve().relative_to(home.resolve())
    if root.stat().st_dev==Path('/projects/u6ada').stat().st_dev:
        raise ValueError('Outputs still share the exhausted project filesystem')
    totals=[]
    for base in (home,root):
        seen=set();total=0
        for directory,dirs,files in os.walk(str(base),followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
            for name in files:
                p=Path(directory)/name
                if p.is_symlink():continue
                try:s=p.stat()
                except FileNotFoundError:continue  # Atomic heartbeat/checkpoint replacement may remove the old name.
                key=(s.st_dev,s.st_ino)
                if key not in seen:total+=max(s.st_size,s.st_blocks*512);seen.add(key)
        totals.append(total)
    if totals[0]>80*1024**3 or totals[1]>32*1024**3:
        raise RuntimeError('Preserve at least20GiB home headroom; bounded run outputs32GiB')
    return dict(home_bytes=totals[0],run_bytes=totals[1],home_limit_GiB=100,run_limit_GiB=32,separate_filesystem=True)


def watch(lease_path,output):
    lease=g.resource(json.loads(Path(lease_path).read_text()));last=None
    while True:
        authority();storage_check()
        last=g.refresh(lease,output,last);time.sleep(10)


def prior_usage(current):
    text=subprocess.check_output(['sacct','-n','-X','-P','-A',g.ACCOUNT,'--starttime=2026-11-08',
        '--format=JobIDRaw,JobName%100,Account,State,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=30)
    rows=[]
    for line in text.splitlines():
        f=line.split('|')
        if len(f)<6 or f[0]==current:continue
        if f[0] in HISTORY:rows.append('|'.join([f[0],f[2],f[3],f[4],f[5]]))
        elif f[1].startswith('duplex-matched') and 'gres/gpu=' in f[5]:
            raise ValueError('Uncounted matched allocation; never reset usage')
    return actual_usage('\n'.join(rows),HISTORY)


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source or a.inputs_sha!=DATA_SHA:raise ValueError('Exact committed execution/data required')
    recovery=authority();storage=storage_check();original=preflight(ORIGINAL_ROOT,PACKET_SHA)
    if g.sha(PRESERVED/'interrupted.pt')!=STATE_SHA:raise ValueError('Preserved actual step3 checkpoint changed')
    jobid=os.environ['SLURM_JOB_ID'];prior=prior_usage(jobid)
    if prior!=3527:raise ValueError('Exact initial quota-recovery carry required')
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',jobid],universal_newlines=True,timeout=20))
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
        continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=3,
        quota_recovery_authorization_sha256=recovery,job_id=jobid,node=job['NodeList'],source=source,
        controller_source=source,worker_worktree=str(Path.cwd()),start_unix=g.stamp(job['StartTime']),
        stop_unix=g.stamp(job['EndTime']),maximum_seconds=int(g.stamp(job['EndTime'])-g.stamp(job['StartTime'])),
        GPU_count=2,prior_GPU_seconds=prior)
    g.validate_job(job,lease,time.time())
    root=DURABLE/'runs'/('quota-recovery-'+jobid);root.mkdir(parents=True,exist_ok=False)
    execution=root/('execution-'+jobid);execution.mkdir()
    write(execution/'lease.json',lease);write(execution/'storage-admission.json',storage)
    write(root/'quota-recovery-initial.json',dict(source=source,authorization_sha256=recovery,failed_job='8235976'))
    # Bind only this private run area, not the entire home or credentials.
    os.environ['APPTAINER_BINDPATH']=str(DURABLE)+':'+str(DURABLE)
    devices=visible_pair(os.environ['CUDA_VISIBLE_DEVICES']);watcher=None;code=1
    try:
        watcher=subprocess.Popen([sys.executable,'-B','-u','-m','training.personaplex.matched_quota_recovery',
            '--watch','--lease',str(execution/'lease.json'),'--output',str(execution/'guard.json')],
            stdout=(execution/'guard.log').open('x'),stderr=subprocess.STDOUT)
        deadline=time.time()+40
        while not (execution/'guard.json').exists():
            if watcher.poll() is not None or time.time()>deadline:raise RuntimeError('Independent storage/Slurm guard failed')
            time.sleep(1)
        children=TrainingChildren(Path.cwd(),source,execution,lease,watcher,
            dict(zip(('personaplex','context-spanning'),devices)),a.inputs,a.inputs_sha)
        entries=initial_plan(original,root,execution)
        row=next(x for x in entries if (x['base'],x['capacity'])==('personaplex','narrow'))
        row.update(checkpoint=str(PRESERVED/'interrupted.pt'),checkpoint_sha256=STATE_SHA,
            quota_interrupted_state='dl-moshi-quota-interrupted-v1',restored_step=3)
        partial=execute(children,entries,root,root/'joint-admission.json',source)
        write(root/'phase-result.json',dict(source=source,all_four_complete=not partial,
            partial_lineages=partial,full_native_behavior_speech_backend_live_required=True,adopted=False));code=0
    finally:
        if watcher is not None:
            watcher.terminate()
            try:watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:watcher.kill();watcher.wait()
        write(execution/'terminal.json',dict(source=source,job_id=jobid,actual_exit_code=code,
            ended_UTC=datetime.now(timezone.utc).isoformat(),adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true')
    for name in ('source','inputs','inputs-sha','lease','output'):p.add_argument('--'+name)
    a=p.parse_args()
    if a.watch:watch(a.lease,a.output)
    else:
        if not all(getattr(a,k) for k in ('source','inputs','inputs_sha')):p.error('Exact source/inputs required')
        main(a)
