"""Prepare failed6 recovery7: exact states, fresh restores, no calibration.

No submission entry exists here. Missing owner amendment fails before any model.
"""
import argparse,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
from . import matched_guard as g,matched_health as health
from .matched_job import write,visible_pair
from .matched_training_only import TrainingChildren,initial_plan,execute
from .matched_reuse import preflight,ORIGINAL_ROOT,DATA_SHA
from .matched_continue_policy import PACKET_SHA
from .matched_storage_job import HISTORY as OLD_HISTORY
from .matched_postupdate import prior_usage
from .matched_failed_state import authority as post_authority
from .matched_normal88 import authority as normal_authority
from .matched_storage_recovery import authority as storage_authority
from .matched_interrupted import DURABLE,AUTH as QUOTA_AUTH
from .matched_storage_quota import storage_check
from . import matched_rpc_recovery as recovery

HISTORY=dict(OLD_HISTORY,**{'8285192':dict(state='FAILED',elapsed=45141,GPUs=2)})



def startup_job(jobid):
    """Read-only retry BEFORE model/lease creation; never retries a submission.

    Three 20s reads at most. No active training freshness deadline is extended.
    Provider StartTime/EndTime remain authoritative and are never reset.
    """
    for attempt in range(3):
        try:
            value=subprocess.check_output(['scontrol','show','job','-o',jobid],
                universal_newlines=True,timeout=20)
            return g.parse_job(value)
        except subprocess.TimeoutExpired:
            print(json.dumps(dict(event='startup_scheduler_timeout',job_id=jobid,
                attempt=attempt+1,maximum_attempts=3)),flush=True)
            if attempt==2:raise


def initial_entries(original,root,execution):
    recovery.authority();rows=initial_plan(original,root,execution)
    for row in rows:
        path,step,digest=recovery.selected(row['base'],row['capacity'])
        reference=json.loads(Path(str(path)+'.native-reference.json').read_text())
        row.update(checkpoint=str(path),checkpoint_sha256=digest,restored_step=step,
            rpc_failure_recovery=recovery.MARKER)
        if g.sha(path)!=digest:raise ValueError('Preserved selected checkpoint changed')
        recovery.validate(row,reference,step)
    return rows


def main(a):
    source=subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()
    if source!=a.source or a.inputs_sha!=DATA_SHA:raise ValueError('Exact recovery producer/data required')
    grant=recovery.authority()  # Must remain before any model, lease or output directory.
    storage=storage_check();original=preflight(ORIGINAL_ROOT,PACKET_SHA)
    jobid=os.environ['SLURM_JOB_ID'];prior=prior_usage(jobid,HISTORY)
    if prior!=134991:raise ValueError('All six actual allocations must remain counted')
    actual=subprocess.check_output(['sacct','-n','-X','-P','-j','8285192',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES'],universal_newlines=True,timeout=20).strip()
    if not actual.startswith('8285192|FAILED|1:0|45141|') or 'gres/gpu=2' not in actual:
        raise ValueError('Actual failed6 must stay failed; no manufactured successful parent')
    job=startup_job(jobid)
    if not job.get('JobName','').startswith('duplex-matched-'):raise ValueError('Exact matched allocation required')
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(g.AUTH),
        continuation_authorization_sha256=g.sha(g.EXTENSION),continuation_interval=7,
        quota_recovery_authorization_sha256=g.sha(QUOTA_AUTH),
        postupdate_recovery_authorization_sha256=post_authority(),
        normal88_authorization_sha256=normal_authority(),storage_recovery_authorization_sha256=storage_authority(),
        rpc_recovery_authorization_sha256=grant,job_id=jobid,node=job['NodeList'],source=source,
        controller_source=source,worker_worktree=str(Path.cwd()),start_unix=g.stamp(job['StartTime']),
        stop_unix=g.stamp(job['EndTime']),maximum_seconds=int(g.stamp(job['EndTime'])-g.stamp(job['StartTime'])),
        GPU_count=2,prior_GPU_seconds=prior)
    g.validate_job(job,lease,time.time());root=DURABLE/'runs'/('rpc-recovery-'+jobid)
    root.mkdir(parents=True,exist_ok=False);execution=root/('execution-'+jobid);execution.mkdir()
    rows=initial_entries(original,root,execution)
    write(execution/'lease.json',lease);write(execution/'storage-admission.json',storage)
    write(root/'rpc-recovery-initial.json',dict(source=source,authorization_sha256=grant,failed_job='8285192'))
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
