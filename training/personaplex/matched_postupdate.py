"""One approved recovery from actual saved89/80 after failed8260542.

No loss/data/precision change, repeated admission, source relabelling or replay.
Host Python3.6; actual workers use the pinned modern native container.
"""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from . import matched_guard as g,matched_health as health
from .matched_job import write,visible_pair
from .matched_training_only import TrainingChildren,initial_plan,execute
from .matched_quota_recovery import HISTORY as OLD_HISTORY,storage_check
from .matched_reuse import preflight,actual_usage,ORIGINAL_ROOT,DATA_SHA
from .matched_continue_policy import PACKET_SHA
from .matched_failed_state import authority,entries,validate_failed,MARKER,SAVED
from .matched_interrupted import DURABLE

HISTORY=dict(OLD_HISTORY,**{'8260542':dict(state='FAILED',elapsed=15147,GPUs=2)})
NORMAL_HISTORY=dict(HISTORY,**{'8273944':dict(state='FAILED',elapsed=119,GPUs=2)})


def prior_usage(current,history=HISTORY):
    text=subprocess.check_output(['sacct','-n','-X','-P','-A',g.ACCOUNT,'--starttime=2026-11-08',
        '--format=JobIDRaw,JobName%100,Account,State,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=30)
    rows=[]
    for line in text.splitlines():
        f=line.split('|')
        if len(f)<6 or f[0]==current:continue
        if f[0] in history:rows.append('|'.join([f[0],f[2],f[3],f[4],f[5]]))
        elif f[1].startswith('duplex-matched') and 'gres/gpu=' in f[5]:
            raise ValueError('Uncounted actual matched allocation; never reset time')
    return actual_usage('\n'.join(rows),history)


def initial_entries(original,root,execution,normal88=False):
    from . import matched_normal88 as normal
    if normal88:normal.authority()
    result=initial_plan(original,root,execution)
    for base,value in entries().items():
        rollback=normal88 and base=='personaplex'
        if rollback:value=dict(value,preserved_checkpoint=str(normal.CHECKPOINT),checkpoint_sha256=normal.SHA,step=88)
        row=next(r for r in result if (r['base'],r['capacity'])==(base,'narrow'))
        row.update(checkpoint=value['preserved_checkpoint'],checkpoint_sha256=value['checkpoint_sha256'],
            restored_step=value['step'])
        row['normal88_recovery' if rollback else 'postupdate_failure_recovery']=normal.MARKER if rollback else MARKER
        path=Path(row['checkpoint']);reference=json.loads(Path(str(path)+'.native-reference.json').read_text())
        if g.sha(path)!=row['checkpoint_sha256']:raise ValueError('Preserved actual postupdate state changed')
        if rollback:normal.validate(row,reference,value['step'])
        else:validate_failed(row,reference,value['step'])
    return result


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source or a.inputs_sha!=DATA_SHA:raise ValueError('Exact committed producer and original qualified inputs required')
    recovery=authority();storage=storage_check();original=preflight(ORIGINAL_ROOT,PACKET_SHA)
    normal88=getattr(a,'normal88',False)
    jobid=os.environ['SLURM_JOB_ID'];prior=prior_usage(jobid,NORMAL_HISTORY if normal88 else HISTORY)
    if prior!=(34059 if normal88 else 33821):raise ValueError('Exact failure-inclusive initial recovery carry required')
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',jobid],universal_newlines=True,timeout=20))
    if not job.get('JobName','').startswith('duplex-matched-'):raise ValueError('Accounted matched job name required')
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
        continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=5 if normal88 else 4,
        quota_recovery_authorization_sha256=g.sha(Path('docs/NATIVE-MOSHI-QUOTA-RECOVERY-AUTHORIZATION-2026-11-09.json')),
        postupdate_recovery_authorization_sha256=recovery,job_id=jobid,node=job['NodeList'],source=source,
        controller_source=source,worker_worktree=str(Path.cwd()),start_unix=g.stamp(job['StartTime']),
        stop_unix=g.stamp(job['EndTime']),maximum_seconds=int(g.stamp(job['EndTime'])-g.stamp(job['StartTime'])),
        GPU_count=2,prior_GPU_seconds=prior)
    if normal88:
        from .matched_normal88 import authority as normal_authority
        lease['normal88_authorization_sha256']=normal_authority()
    g.validate_job(job,lease,time.time())
    root=DURABLE/'runs'/(('normal88-recovery-' if normal88 else 'postupdate-recovery-')+jobid);root.mkdir(parents=True,exist_ok=False)
    execution=root/('execution-'+jobid);execution.mkdir()
    rows=initial_entries(original,root,execution,normal88)
    write(execution/'lease.json',lease);write(execution/'storage-admission.json',storage)
    if normal88:write(root/'normal88-recovery-initial.json',dict(source=source,authorization_sha256=lease['normal88_authorization_sha256'],failed_job='8273944'))
    else:write(root/'postupdate-recovery-initial.json',dict(source=source,authorization_sha256=recovery,failed_job='8260542'))
    devices=visible_pair(os.environ['CUDA_VISIBLE_DEVICES']);services=[];control=None;code=1
    try:
        watcher,services,control=health.start(lease,execution)
        children=TrainingChildren(Path.cwd(),source,execution,lease,watcher,
            dict(zip(('personaplex','context-spanning'),devices)),a.inputs,a.inputs_sha)
        children.guard_path=control/'guard.json'
        partial=execute(children,rows,root,root/'joint-admission.json',source)
        write(root/'phase-result.json',dict(source=source,all_four_complete=not partial,
            partial_lineages=partial,full_native_behavior_speech_backend_live_required=True,adopted=False));code=0
    finally:
        if control is not None:health.stop(services,control,execution)
        write(execution/'terminal.json',dict(source=source,job_id=jobid,actual_exit_code=code,
            ended_UTC=datetime.now(timezone.utc).isoformat(),adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--normal88',action='store_true')
    for name in ('source','inputs','inputs-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
