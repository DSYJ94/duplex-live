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
REFERENCE_AUTH=Path('docs/NATIVE-LOCAL-REASONER-REFERENCE-2026-11-16.json')
OFFICIAL_AUTH=Path('docs/NATIVE-OFFICIAL-PERSONAPLEX-BASELINE-2026-11-16.json')
FEATURE_AUTH=Path('docs/NATIVE-GPU-FEATURE-RESTORE-2026-11-17.json')
RESTORE_AUTH=Path('docs/NATIVE-GPU-RESTORE-ONLY-2026-11-17.json')
PAIR_AUTH=Path('docs/NATIVE-RET-PAIRED-EXECUTION-2026-11-17.json')
PAIR_RECOVERY_AUTH=Path('docs/NATIVE-RET-PAIRED-RECOVERY-2026-11-17.json')
TRAIN_FAILURE_AUTH=Path('docs/NATIVE-TRAIN1008-FAILURE-COLLECTION-2026-11-17.json')
CORPUS_REVIEW_AUTH=Path('docs/NATIVE-BROAD-CORPUS-REVIEW-2026-11-17.json')
CORPUS_REPAIR_AUTH=Path('docs/NATIVE-BROAD-CORPUS-REVIEW-REPAIR-2026-11-17.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resource(lease):
    grants=[(k,p,status,seconds)for k,p,status,seconds in (
        ('reference_authorization_sha256',REFERENCE_AUTH,'OWNER_DIRECTED_FROZEN_LOCAL_REFERENCE',7200),
        ('official_authorization_sha256',OFFICIAL_AUTH,'OWNER_DIRECTED_OFFICIAL_CONTRACT_BASELINE',18000),
        ('feature_authorization_sha256',FEATURE_AUTH,'OWNER_DIRECTED_NATIVE_FEATURE_RESTORE',21600),
        ('restore_authorization_sha256',RESTORE_AUTH,'OWNER_DIRECTED_NATIVE_RESTORE_ONLY',16200))if k in lease]
    if len(grants)>1:raise ValueError('No mixed inference scopes')
    paired='paired_authorization_sha256'in lease
    if paired and grants:raise ValueError('No mixing learned and zero-update scopes')
    count=4;maximum=86400
    if 'corpus_review_authorization_sha256' in lease:
        scope=json.loads(CORPUS_REVIEW_AUTH.read_text())
        if (paired or grants or 'train_failure_authorization_sha256' in lease
                or lease['corpus_review_authorization_sha256']!=sha(CORPUS_REVIEW_AUTH)
                or scope.get('status')!='GOAL_DIRECTED_FROZEN_PUBLIC_CORPUS_REVIEW'
                or scope.get('GPU_count')!=4 or scope.get('maximum_seconds')!=43200
                or scope.get('required_candidates')!=17276 or scope.get('optimizer_updates')!=0
                or scope.get('external_API_calls')!=0 or scope.get('automatic_retries')!=0
                or lease.get('corpus_review_parent_job')!='8553179'):
            raise ValueError('Exact separate frozen whole-corpus review scope required')
        count=4;maximum=43200
    if 'corpus_review_repair_sha256' in lease:
        scope=json.loads(CORPUS_REPAIR_AUTH.read_text())
        if ('corpus_review_authorization_sha256' not in lease
                or lease['corpus_review_repair_sha256']!=sha(CORPUS_REPAIR_AUTH)
                or scope.get('status')!='GOAL_DIRECTED_CORPUS_REVIEW_FORMAT_REPAIR'
                or scope.get('parent_job')!='8555523' or scope.get('GPU_count')!=4
                or scope.get('parent_elapsed_seconds_upper')!=600 or scope.get('maximum_seconds')!=42600
                or scope.get('maximum_combined_wall_seconds')!=43200 or scope.get('batch_size')!=4
                or scope.get('required_candidates')!=17276 or scope.get('optimizer_updates')!=0
                or scope.get('external_API_calls')!=0 or scope.get('automatic_retries')!=0):
            raise ValueError('Exact unused-time corpus review repair required')
        maximum=42600
    if 'train_failure_authorization_sha256' in lease:
        scope=json.loads(TRAIN_FAILURE_AUTH.read_text())
        if (paired or grants or lease['train_failure_authorization_sha256']!=sha(TRAIN_FAILURE_AUTH)
                or scope.get('status')!='GOAL_DIRECTED_TRAIN_FAILURE_COLLECTION'
                or scope.get('GPU_count')!=4 or scope.get('maximum_seconds')!=43200
                or scope.get('required_TRAIN_outputs')!=1008 or scope.get('DEV_inputs')!=0
                or scope.get('optimizer_updates')!=0 or scope.get('external_API_calls')!=0
                or scope.get('automatic_retries')!=0):raise ValueError('Exact separate TRAIN-only collection scope required')
        count=4;maximum=43200
    if paired:
        scope=json.loads(PAIR_AUTH.read_text())
        if (lease['paired_authorization_sha256']!=sha(PAIR_AUTH)or scope['status']!='OWNER_AUTHORIZED_PAIRED_RET'
                or scope['GPU_count']!=1 or scope['maximum_seconds']!=43200
                or scope['maximum_new_readout_updates']!=256 or scope['maximum_updates_per_arm']!=128
                or scope['native_backbone_updates']!=0 or scope['external_API_calls']!=0
                or scope['arms']!=['readout','bias']):raise ValueError('Separate authorized paired readout scope required')
        count=1;maximum=43200
    if 'paired_recovery_sha256'in lease:
        recovery=json.loads(PAIR_RECOVERY_AUTH.read_text())
        if (not paired or lease['paired_recovery_sha256']!=sha(PAIR_RECOVERY_AUTH)
                or recovery['status']!='GOAL_DIRECTED_PAIRED_AUDIT_RECOVERY'
                or recovery['parent_actual_GPU_seconds']!=5394 or recovery['maximum_seconds']!=37800
                or recovery['parent_actual_GPU_seconds']+recovery['maximum_seconds']>43200
                or recovery['remaining_bias_updates']!=128 or recovery['readout_updates_repeated']!=0
                or recovery['readout_generations_repeated']!=0 or recovery['external_API_calls']!=0
                or recovery['native_backbone_updates']!=0 or lease.get('paired_parent_job')!='8540802'
                or lease.get('prior_phase_GPU_seconds')!=5394):raise ValueError('Exact remaining paired recovery bound required')
        maximum=37800
    if grants:
        key,path,status,maximum=grants[0];scope=json.loads(path.read_text())
        if (lease[key]!=sha(path)or scope['status']!=status or scope['GPU_count']!=1 or scope['maximum_seconds']!=maximum
                or scope['external_API_calls']!=0 or scope['optimizer_updates']!=0):raise ValueError('Exact separately declared frozen inference scope required')
        count=1
    scope=json.loads(AUTH.read_text())
    if scope['status']!='OWNER_AUTHORIZED_CONTINUATION' or scope['award_total_GPU_hours']!=20000 or scope['maximum_simultaneous_voice_GPUs']!=4:raise ValueError('Exact current owner continuation scope required')
    if lease.get('provider')!='Slurm' or lease.get('resource_key')!=KEY or lease.get('account')!=ACCOUNT or lease.get('authorization_sha256')!=sha(AUTH) or lease.get('GPU_count')!=count or not str(lease.get('job_id','')).isdigit():raise ValueError('Exact declared scheduler allocation required')
    if not 0<lease['stop_unix']-lease['start_unix']<=maximum:raise ValueError('Immutable single-job scheduler bound required')
    prior=lease['prior_award_GPU_seconds'];other=lease['other_reserved_GPU_seconds']
    if not all(isinstance(x,(int,float)) and x>=0 for x in (prior,other)) or prior+other+count*(lease['stop_unix']-lease['start_unix'])>20000*3600:raise ValueError('All prior usage and concurrent reservations must fit original award')
    return lease


def validate_job(job,lease,now):
    resource(lease);tres=dict(x.split('=',1)for x in job.get('AllocTRES','').split(',')if '='in x)
    if job.get('JobId')!=lease['job_id'] or job.get('Account')!=ACCOUNT or job.get('JobState')!='RUNNING' or tres.get('gres/gpu')!=str(lease['GPU_count']) or job.get('NumNodes')not in ('1','1-1') or job.get('NodeList')!=lease['node'] or job.get('Partition')!='workq':raise ValueError('Actual exact running allocated node required')
    if stamp(job['StartTime'])!=lease['start_unix'] or stamp(job['EndTime'])!=lease['stop_unix'] or not lease['start_unix']<=now<lease['stop_unix']-180:raise ValueError('Actual original provider clock differs')


def ensure(lease,guard,needed=0,*,now=None,environ=None):
    resource(lease);now=time.time()if now is None else now;env=os.environ if environ is None else environ
    if env.get('SLURM_JOB_ID')!=lease['job_id'] or env.get('SLURM_JOB_ACCOUNT')!=ACCOUNT:raise RuntimeError('Worker must remain inside its own allocation')
    for key in ('job_id','account','node','source','authorization_sha256','GPU_count','start_unix','stop_unix'):
        if lease[key]!=guard.get(key):raise RuntimeError('Independent guard identity changed: '+key)
    if not guard.get('provider_job_verified') or guard.get('terminal') or not 0<=now-guard['heartbeat_unix']<=45 or needed<0:raise RuntimeError('Actual guard missing, terminal, expired or insufficient preservation time')
    if now+needed>=lease['stop_unix']-180:raise TimeoutError('Immutable scheduler deadline requires checkpoint continuation')


def check(path,needed=0):
    lease=resource(json.loads(Path(path).read_text()))
    if subprocess.check_output(['git','rev-parse','HEAD'],universal_newlines=True).strip()!=lease['source']:raise RuntimeError('Actual committed worker source differs from admitted lease')
    guard=json.loads(Path(os.environ['DUPLEX_LIVE_GUARD_FILE']).read_text());ensure(lease,guard,needed);return lease


def watch(path,output):
    lease=resource(json.loads(Path(path).read_text()));last_query=None;verified=False;terminal=False;observation={}
    output=Path(output)
    while time.time()<lease['stop_unix']-30:
        now=time.time()
        if last_query is None or now-last_query>=60:
            last_query=now
            job=None
            try:
                raw=subprocess.check_output(['scontrol','show','job','-o',lease['job_id']],universal_newlines=True,timeout=20);job=parse_job(raw)
                if verified and job.get('JobId')==lease['job_id'] and job.get('JobState')in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'):terminal=True
                else:validate_job(job,lease,time.time());verified=True
                observation=dict(provider_checked_unix=time.time(),provider_state=job.get('JobState'),observation_error=None)
            except Exception as exc:
                required={'JobId','Account','JobState','AllocTRES','NumNodes','NodeList','Partition','StartTime','EndTime'}
                if verified and job is not None and required<=set(job) and job.get('JobState')=='RUNNING':terminal=True
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
