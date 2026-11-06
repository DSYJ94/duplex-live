"""Real Slurm identity/deadline guard, independent of Mac/cloud credentials.

The scheduler enforces the job walltime; a separate host process refreshes this
short lease. Native code checks it before loading weights and every voice step.
Compatible with the host's Python3.6 as well as the container's Python3.12.
"""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,subprocess,time
from pathlib import Path

KEY='venus-isambard-v1'
ACCOUNT='brics.u6ada'
AUTH=Path('docs/NATIVE-ISAMBARD-TRAINING-CONTINUATION-2026-11-06.json')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def parse_job(text):
    return dict(x.split('=',1) for x in text.split() if '=' in x)

def stamp(value):
    return datetime.strptime(value,'%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).timestamp()

def resource(lease):
    a=json.loads(AUTH.read_text())
    if a.get('status')!='OWNER_INSTRUCTED_CONTINUE_THROUGH_TRAINING' or a.get('provider')!='Isambard' or a.get('account')!=ACCOUNT:
        raise ValueError('Explicit Isambard continuation required')
    if lease.get('provider')!='Slurm' or lease.get('resource_key')!=KEY or lease.get('account')!=ACCOUNT or lease.get('authorization_sha256')!=sha(AUTH):
        raise ValueError('Exact Slurm authority required; never a synthetic GCP lease')
    if not str(lease.get('job_id','')).isdigit() or not 0<lease.get('maximum_seconds',0)<=86400 or lease.get('GPU_count')!=1:
        raise ValueError('Bounded one-GPU scheduler allocation required')
    return lease

def validate_job(job,lease,now):
    resource(lease)
    tres=dict(x.split('=',1) for x in job.get('AllocTRES','').split(',') if '=' in x)
    if job.get('JobId')!=lease['job_id'] or job.get('Account')!=ACCOUNT or job.get('JobState')!='RUNNING' or tres.get('gres/gpu')!='1':
        raise ValueError('Actual running project/oneGPU Slurm identity differs')
    if job.get('NumNodes') not in ('1','1-1') or job.get('NodeList')!=lease['node'] or job.get('Partition') not in ('interactive','workq'):
        raise ValueError('Actual scheduler node/partition differs')
    start,end=stamp(job['StartTime']),stamp(job['EndTime'])
    if start!=lease['start_unix'] or lease['stop_unix']>end or end-start>lease['maximum_seconds'] or not start<=now<lease['stop_unix']-30:
        raise ValueError('Actual scheduler clock/deadline differs or expired')

def ensure(lease,guard,now=None,environ=None):
    resource(lease);now=time.time() if now is None else now;env=os.environ if environ is None else environ
    if env.get('SLURM_JOB_ID')!=lease['job_id'] or env.get('SLURM_JOB_ACCOUNT')!=ACCOUNT:
        raise RuntimeError('Native work must stay inside its actual scheduler allocation')
    for key in ('job_id','account','node','source','authorization_sha256','stop_unix'):
        if guard.get(key)!=lease.get(key):raise RuntimeError('Independent Slurm guard binding differs: '+key)
    if not 0<=now-guard['checked_unix']<=45 or not now<min(guard['expires_unix'],guard['stop_unix']-30):
        raise RuntimeError('Independent Slurm guard is absent or expired')

def watch(lease_path,output):
    lease=json.loads(Path(lease_path).read_text());resource(lease)
    while True:
        text=subprocess.check_output(['scontrol','show','job','-o',lease['job_id']],universal_newlines=True,timeout=20)
        now=time.time();validate_job(parse_job(text),lease,now)
        row=dict((k,lease[k]) for k in ('job_id','account','node','source','authorization_sha256','stop_unix'))
        row.update(checked_unix=now,expires_unix=min(now+45,lease['stop_unix']-30))
        path=Path(output);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row));os.replace(str(tmp),str(path))
        time.sleep(10)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--lease',required=True);p.add_argument('--output',required=True);a=p.parse_args();watch(a.lease,a.output)
