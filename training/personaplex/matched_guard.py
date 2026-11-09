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


def resource(lease):
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
            or type(lease.get('prior_GPU_seconds')) is not int or not 327<=lease['prior_GPU_seconds']<LIMIT
            or type(lease.get('maximum_seconds')) is not int or not 0<lease['maximum_seconds']<=86400
            or lease['prior_GPU_seconds']+2*lease['maximum_seconds']>LIMIT):
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
            or lease['prior_GPU_seconds']+2*(end-start)>LIMIT):
        raise ValueError('Original scheduler deadline or aggregate budget differs')


def ensure(lease,guard,now=None,environ=None,needed=0):
    resource(lease)
    now=time.time() if now is None else now
    env=os.environ if environ is None else environ
    if env.get('SLURM_JOB_ID')!=lease['job_id'] or env.get('SLURM_JOB_ACCOUNT')!=ACCOUNT:
        raise RuntimeError('Native learner must remain inside its scheduler allocation')
    for key in ('job_id','account','node','source','authorization_sha256','stop_unix','GPU_count','prior_GPU_seconds'):
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
    now=time.time();deadline=now+20 if last_success is None else min(now+20,last_success+44)
    try:
        job=subprocess.check_output(['scontrol','show','job','-o',lease['job_id']],
            universal_newlines=True,timeout=max(1,deadline-time.time()))
        queue=subprocess.check_output(['squeue','-A',ACCOUNT,'-h','-o','%i'],
            universal_newlines=True,timeout=max(1,deadline-time.time()))
    except subprocess.TimeoutExpired:
        if last_success is None or time.time()>=last_success+44:raise
        print(json.dumps(dict(event='observation_timeout',original_success_unix=last_success)),flush=True)
        return last_success
    if set(queue.split())!={lease['job_id']}:
        raise RuntimeError('Another project job is present; no overlapping submissions')
    now=time.time();validate_job(parse_job(job),lease,now)
    row=dict((k,lease[k]) for k in ('job_id','account','node','source','authorization_sha256','stop_unix','GPU_count','prior_GPU_seconds'))
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
