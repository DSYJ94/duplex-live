"""Immutable Slurm bounds for the new expanded-data phase, separate from old grants.

Unknown scheduler observations do not cancel valid admitted training. They also
cannot admit a new job, extend its deadline, or grant additional GPU capacity.
"""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
from .matched_guard import parse_job,stamp

AUTH=Path('docs/NATIVE-EXPANDED1072-EXECUTION-2026-11-14.json')
ACCOUNT='brics.u6ada'
KEY='duplex-expanded1072-v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource(lease):
    scope=json.loads(AUTH.read_text())
    if scope['status']!='OWNER_AUTHORIZED_CONTINUATION' or scope['award_total_GPU_hours']!=20000 or scope['maximum_simultaneous_voice_GPUs']!=4:raise ValueError('Exact current owner continuation scope required')
    if lease.get('provider')!='Slurm' or lease.get('resource_key')!=KEY or lease.get('account')!=ACCOUNT or lease.get('authorization_sha256')!=sha(AUTH) or lease.get('GPU_count')!=4 or not str(lease.get('job_id','')).isdigit():raise ValueError('Exact declared four-GPU scheduler allocation required')
    if not 0<lease['stop_unix']-lease['start_unix']<=86400:raise ValueError('Immutable single-job scheduler bound required')
    prior=lease['prior_award_GPU_seconds'];other=lease['other_reserved_GPU_seconds']
    if not all(isinstance(x,(int,float)) and x>=0 for x in (prior,other)) or prior+other+4*(lease['stop_unix']-lease['start_unix'])>20000*3600:raise ValueError('All prior usage and concurrent reservations must fit original award')
    return lease


def validate_job(job,lease,now):
    resource(lease);tres=dict(x.split('=',1)for x in job.get('AllocTRES','').split(',')if '='in x)
    if job.get('JobId')!=lease['job_id'] or job.get('Account')!=ACCOUNT or job.get('JobState')!='RUNNING' or tres.get('gres/gpu')!='4' or job.get('NumNodes')not in ('1','1-1') or job.get('NodeList')!=lease['node'] or job.get('Partition')!='workq':raise ValueError('Actual exact running four-GPU node required')
    if stamp(job['StartTime'])!=lease['start_unix'] or stamp(job['EndTime'])!=lease['stop_unix'] or not lease['start_unix']<=now<lease['stop_unix']-180:raise ValueError('Actual original provider clock differs')


def ensure(lease,guard,needed=0,*,now=None,environ=None):
    resource(lease);now=time.time()if now is None else now;env=os.environ if environ is None else environ
    if env.get('SLURM_JOB_ID')!=lease['job_id'] or env.get('SLURM_JOB_ACCOUNT')!=ACCOUNT:raise RuntimeError('Worker must remain inside its own allocation')
    for key in ('job_id','account','node','source','authorization_sha256','GPU_count','start_unix','stop_unix'):
        if lease[key]!=guard.get(key):raise RuntimeError('Independent guard identity changed: '+key)
    if not guard.get('provider_job_verified') or guard.get('terminal') or not 0<=now-guard['heartbeat_unix']<=45 or needed<0 or now+needed>=lease['stop_unix']-180:raise RuntimeError('Actual guard missing, terminal, expired or insufficient preservation time')


def check(path,needed=0):
    lease=resource(json.loads(Path(path).read_text()));guard=json.loads(Path(os.environ['DUPLEX_LIVE_GUARD_FILE']).read_text());ensure(lease,guard,needed);return lease


def watch(path,output):
    lease=resource(json.loads(Path(path).read_text()));last_query=None;verified=False;terminal=False;observation={}
    output=Path(output)
    while time.time()<lease['stop_unix']-30:
        now=time.time()
        if last_query is None or now-last_query>=60:
            last_query=now
            try:
                raw=subprocess.check_output(['scontrol','show','job','-o',lease['job_id']],universal_newlines=True,timeout=20);job=parse_job(raw)
                if verified and job.get('JobId')==lease['job_id'] and job.get('JobState')in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):terminal=True
                else:validate_job(job,lease,time.time());verified=True
                observation=dict(provider_checked_unix=time.time(),provider_state=job.get('JobState'),observation_error=None)
            except Exception as exc:
                observation=dict(observation,observation_error=type(exc).__name__)
                # Initial admission still requires a real provider success.
                # After admission, immutable walltime remains enforced while
                # transient read errors are recorded without killing learning.
                print(json.dumps(dict(event='provider_observation_unavailable',admitted=verified,error=type(exc).__name__)),flush=True)
        if verified:
            row={k:lease[k]for k in ('job_id','account','node','source','authorization_sha256','GPU_count','start_unix','stop_unix')};row.update(observation,heartbeat_unix=time.time(),provider_job_verified=True,terminal=terminal)
            temporary=output.with_suffix('.tmp');temporary.write_text(json.dumps(row)+'\n');os.replace(temporary,output)
        if terminal:return
        time.sleep(10)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--lease',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();watch(a.lease,a.output)
