"""Separate real-Slurm guard for the explicitly approved two-GPU comparison.

The old one-GPU guard remains unchanged. Scheduler walltime and a separate host
watcher enforce the same immutable aggregate budget, without Mac credentials.
This module is also compatible with the login/host Python3.6.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
from pathlib import Path

KEY='moshi-matched-two-gpu-v1'
ACCOUNT='brics.u6ada'
AUTH=Path('docs/NATIVE-MOSHI-MATCHED-TRAINING-AUTHORIZATION-2026-11-08.json')
LIMIT=96*3600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_job(text):
    return dict(x.split('=',1) for x in text.split() if '=' in x)


def stamp(value):
    return datetime.strptime(value,'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()


EXTENSION=Path('docs/NATIVE-MOSHI-MATCHED-TRAINING-ONLY-AUTHORIZATION-2026-11-09.json')


def phase_limit(lease):
    digest=lease.get('continuation_authorization_sha256')
    if digest is None:return LIMIT
    authority=json.loads(EXTENSION.read_text())
    required=dict(status='OWNER_APPROVED_MATCHED_TRAINING_ONLY_CONTINUATION',
        proposal_commit='298521a76959a9d4f60c85180456944d37a27ca6',provider='Isambard',account=ACCOUNT,
        maximum_aggregate_GPU_hours=112,maximum_single_job_seconds=86400,simultaneous_GPUs=2,
        original_admission_source='8b822b0d85ccea4f291255204cbc358fa85469c4',
        new_external_model_API_calls=0,new_purchases=False,new_GCP_compute=False,
        repeat_native_calibration=False,maximum_future_running_intervals=4)
    intervals=4
    if lease.get('postupdate_recovery_authorization_sha256') is not None:
        from .matched_failed_state import authority as recovery_authority
        if lease['postupdate_recovery_authorization_sha256']!=recovery_authority():
            raise ValueError('Actual postupdate recovery authorization differs')
        intervals=6
    if lease.get('normal88_authorization_sha256') is not None:
        from .matched_normal88 import authority as normal_authority
        if lease['normal88_authorization_sha256']!=normal_authority() or intervals!=6:
            raise ValueError('Exact paired rollback and postupdate authorities required')
    if lease.get('storage_recovery_authorization_sha256') is not None:
        from .matched_storage_recovery import authority as storage_authority
        if lease['storage_recovery_authorization_sha256']!=storage_authority() or intervals!=6:
            raise ValueError('Exact storage recovery and all earlier authorities required')
        intervals=8
    if type(lease.get('continuation_interval')) is not int or not 1<=lease['continuation_interval']<=intervals:
        raise ValueError('Bounded actual continuation interval required')
    if sha(EXTENSION)!=digest or any(authority.get(k)!=v for k,v in required.items()):
        raise ValueError('Exact owner-approved training-only amendment required')
    return 112*3600


def resource(lease):
    limit=phase_limit(lease)
    authority=json.loads(AUTH.read_text())
    if (authority.get('status')!='OWNER_SELECTED_MATCHED_TRAINING_COMPARISON'
            or authority.get('provider')!='Isambard' or authority.get('account')!=ACCOUNT
            or authority.get('simultaneous_GPUs')!=2 or authority.get('maximum_aggregate_GPU_hours')!=96
            or authority.get('maximum_single_job_seconds')!=86400
            or authority.get('new_external_model_API_calls')!=0):
        raise ValueError('Exact approved matched comparison required')
    if (lease.get('provider')!='Slurm' or lease.get('resource_key')!=KEY
            or lease.get('account')!=ACCOUNT or lease.get('authorization_sha256')!=sha(AUTH)
            or lease.get('GPU_count')!=2 or not str(lease.get('job_id','')).isdigit()
            or type(lease.get('prior_GPU_seconds')) is not int or not 327<=lease['prior_GPU_seconds']<limit
            or type(lease.get('maximum_seconds')) is not int or not 0<lease['maximum_seconds']<=86400
            or lease['prior_GPU_seconds']+2*lease['maximum_seconds']>limit):
        raise ValueError('Bounded actual two-GPU lease with carried allocation time required')
    return lease


def validate_job(job,lease,now):
    resource(lease)
    tres=dict(x.split('=',1) for x in job.get('AllocTRES','').split(',') if '=' in x)
    if (job.get('JobId')!=lease['job_id'] or job.get('Account')!=ACCOUNT
            or job.get('JobState')!='RUNNING' or tres.get('gres/gpu')!='2'
            or job.get('NumNodes') not in ('1','1-1') or job.get('NodeList')!=lease['node']
            or job.get('Partition') not in ('interactive','workq')):
        raise ValueError('Exact actual running one-node/two-GPU scheduler identity required')
    start,end=stamp(job['StartTime']),stamp(job['EndTime'])
    if (start!=lease['start_unix'] or lease['stop_unix']!=end
            or end-start>lease['maximum_seconds'] or not start<=now<end-30
            or lease['prior_GPU_seconds']+2*(end-start)>phase_limit(lease)):
        raise ValueError('Original scheduler deadline or aggregate budget differs')


def ensure(lease,guard,now=None,environ=None,needed=0):
    resource(lease)
    now=time.time() if now is None else now
    env=os.environ if environ is None else environ
    if env.get('SLURM_JOB_ID')!=lease['job_id'] or env.get('SLURM_JOB_ACCOUNT')!=ACCOUNT:
        raise RuntimeError('Native learner must remain inside its scheduler allocation')
    for key in ('job_id','account','node','source','authorization_sha256','stop_unix','GPU_count','prior_GPU_seconds','continuation_authorization_sha256','continuation_interval','postupdate_recovery_authorization_sha256','normal88_authorization_sha256','storage_recovery_authorization_sha256'):
        if guard.get(key)!=lease.get(key):raise RuntimeError('Independent matched guard identity differs: '+key)
    if (not 0<=now-guard['checked_unix']<=45
            or not now<min(guard['expires_unix'],guard['stop_unix']-30)
            or needed<0 or now+needed>=lease['stop_unix']-180):
        raise RuntimeError('Independent guard absent/expired or insufficient state-preservation time')


def check(path,needed=0):
    lease=resource(json.loads(Path(path).read_text()))
    guard=json.loads(Path(os.environ['DUPLEX_LIVE_GUARD_FILE']).read_text())
    ensure(lease,guard,needed=needed)
    return lease


def refresh(lease,output,last_success=None):
    from concurrent.futures import ThreadPoolExecutor
    now=time.time();deadline=now+20 if last_success is None else min(now+20,last_success+44)
    def query(command):
        return subprocess.check_output(command,universal_newlines=True,
            timeout=max(1,deadline-time.time()))
    try:
        # Same two authoritative reads and shared deadline; neither read spends
        # the other's response budget. Failed/partial pairs never renew a guard.
        with ThreadPoolExecutor(max_workers=2) as pool:
            job_future=pool.submit(query,['scontrol','show','job','-o',lease['job_id']])
            queue_future=pool.submit(query,['squeue','-A',ACCOUNT,'-h','-o','%i'])
            job=job_future.result();queue=queue_future.result()
    except subprocess.TimeoutExpired:
        if last_success is None or time.time()>=last_success+44:raise
        print(json.dumps(dict(event='observation_timeout',original_success_unix=last_success)),flush=True)
        return last_success
    if set(queue.split())!={lease['job_id']}:
        raise RuntimeError('Another project job is present; no overlapping submissions')
    now=time.time();validate_job(parse_job(job),lease,now)
    row=dict((k,lease[k]) for k in ('job_id','account','node','source','authorization_sha256','stop_unix','GPU_count','prior_GPU_seconds'))
    if lease.get('continuation_authorization_sha256') is not None:
        row['continuation_authorization_sha256']=lease['continuation_authorization_sha256']
        row['continuation_interval']=lease['continuation_interval']
    if lease.get('postupdate_recovery_authorization_sha256') is not None:
        row['postupdate_recovery_authorization_sha256']=lease['postupdate_recovery_authorization_sha256']
    if lease.get('normal88_authorization_sha256') is not None:
        row['normal88_authorization_sha256']=lease['normal88_authorization_sha256']
    if lease.get('storage_recovery_authorization_sha256') is not None:
        row['storage_recovery_authorization_sha256']=lease['storage_recovery_authorization_sha256']
    row.update(checked_unix=now,expires_unix=min(now+45,lease['stop_unix']-30))
    path=Path(output);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row));os.replace(str(tmp),str(path))
    return now


def watch(lease_path,output):
    lease=resource(json.loads(Path(lease_path).read_text()));last=None
    while True:
        last=refresh(lease,output,last);time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lease',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();watch(a.lease,a.output)
