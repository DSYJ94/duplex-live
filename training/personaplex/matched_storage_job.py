"""One approved start6 from normal112/104; immediate learning after four restores.

No numerical admission, changed loss/data, scheduler submission or blind retry.
Host Python3.6; native workers use the original pinned modern runtime/math.
"""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from . import matched_guard as g,matched_health as health
from .matched_job import write,visible_pair
from .matched_training_only import TrainingChildren,initial_plan,execute
from .matched_reuse import preflight,ORIGINAL_ROOT,DATA_SHA
from .matched_continue_policy import PACKET_SHA
from .matched_postupdate import NORMAL_HISTORY,prior_usage
from .matched_failed_state import authority as post_authority
from .matched_normal88 import authority as normal_authority
from .matched_interrupted import DURABLE,AUTH as QUOTA_AUTH
from .matched_storage_recovery import authority,STATES,FAILED_ROOT,MARKER,validate
from .matched_storage_quota import storage_check

HISTORY=dict(NORMAL_HISTORY,**{'8274714':dict(state='FAILED',elapsed=5325,GPUs=2)})


def initial_entries(original,root,execution):
    authority();result=initial_plan(original,root,execution)
    for base,(step,digest) in STATES.items():
        row=next(r for r in result if (r['base'],r['capacity'])==(base,'narrow'))
        path=FAILED_ROOT/base/'narrow/train/live.pt'
        row.update(checkpoint=str(path),checkpoint_sha256=digest,restored_step=step,storage_recovery=MARKER)
        reference=json.loads(Path(str(path)+'.native-reference.json').read_text())
        if g.sha(path)!=digest:raise ValueError('Actual selected normal checkpoint changed')
        validate(row,reference,step)
    return result


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source or a.inputs_sha!=DATA_SHA:raise ValueError('Exact committed worker and original data required')
    grant=authority();storage=storage_check();original=preflight(ORIGINAL_ROOT,PACKET_SHA)
    jobid=os.environ['SLURM_JOB_ID'];prior=prior_usage(jobid,HISTORY)
    if prior!=44709:raise ValueError('Every actual failed GPU interval must remain counted')
    failed=subprocess.check_output(['sacct','-n','-X','-P','-j','8274714',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=20).strip()
    if not failed.startswith('8274714|FAILED|1:0|5325|') or 'gres/gpu=2' not in failed:
        raise ValueError('Exact actual failed window5 required; never relabel success')
    job=g.parse_job(subprocess.check_output(['scontrol','show','job','-o',jobid],universal_newlines=True,timeout=20))
    if not job.get('JobName','').startswith('duplex-matched-'):raise ValueError('Accounted matched allocation required')
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
        continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=6,
        quota_recovery_authorization_sha256=g.sha(QUOTA_AUTH),postupdate_recovery_authorization_sha256=post_authority(),
        normal88_authorization_sha256=normal_authority(),storage_recovery_authorization_sha256=grant,
        job_id=jobid,node=job['NodeList'],source=source,controller_source=source,worker_worktree=str(Path.cwd()),
        start_unix=g.stamp(job['StartTime']),stop_unix=g.stamp(job['EndTime']),
        maximum_seconds=int(g.stamp(job['EndTime'])-g.stamp(job['StartTime'])),GPU_count=2,prior_GPU_seconds=prior)
    g.validate_job(job,lease,time.time());root=DURABLE/'runs'/('storage-recovery-'+jobid)
    root.mkdir(parents=True,exist_ok=False);execution=root/('execution-'+jobid);execution.mkdir()
    rows=initial_entries(original,root,execution);write(execution/'lease.json',lease);write(execution/'storage-admission.json',storage)
    write(root/'storage-recovery-initial.json',dict(source=source,authorization_sha256=grant,failed_job='8274714'))
    devices=visible_pair(os.environ['CUDA_VISIBLE_DEVICES']);services=[];control=None;code=1
    try:
        watcher,services,control=health.start(lease,execution)
        children=TrainingChildren(Path.cwd(),source,execution,lease,watcher,
            dict(zip(('personaplex','context-spanning'),devices)),a.inputs,a.inputs_sha)
        children.guard_path=control/'guard.json'
        partial=execute(children,rows,root,root/'joint-admission.json',source)
        write(root/'phase-result.json',dict(source=source,all_four_complete=not partial,partial_lineages=partial,
            full_native_behavior_speech_backend_live_required=True,adopted=False));code=0
    finally:
        if control is not None:health.stop(services,control,execution)
        write(execution/'terminal.json',dict(source=source,job_id=jobid,actual_exit_code=code,
            ended_UTC=datetime.now(timezone.utc).isoformat(),adopted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','inputs','inputs-sha'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
