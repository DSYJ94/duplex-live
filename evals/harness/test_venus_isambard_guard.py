import copy,json
from pathlib import Path
import pytest
from training.venus import isambard_guard as g

@pytest.fixture
def bound(tmp_path,monkeypatch):
    auth=tmp_path/'authority.json';auth.write_text(json.dumps(dict(status='OWNER_INSTRUCTED_CONTINUE_THROUGH_TRAINING',provider='Isambard',account=g.ACCOUNT)))
    monkeypatch.setattr(g,'AUTH',auth)
    start=g.stamp('2026-11-06T18:00:00')
    lease=dict(provider='Slurm',resource_key=g.KEY,account=g.ACCOUNT,authorization_sha256=g.sha(auth),job_id='123',node='nid1',source='fixed-source',start_unix=start,stop_unix=start+3600,maximum_seconds=3600,GPU_count=1)
    job=g.parse_job('JobId=123 Account=brics.u6ada JobState=RUNNING AllocTRES=cpu=72,gres/gpu=1 NumNodes=1 NodeList=nid1 Partition=interactive StartTime=2026-11-06T18:00:00 EndTime=2026-11-06T19:00:00')
    now=start+100
    guard={k:lease[k] for k in ('job_id','account','node','source','authorization_sha256','stop_unix')};guard.update(checked_unix=now,expires_unix=now+45)
    env=dict(SLURM_JOB_ID='123',SLURM_JOB_ACCOUNT=g.ACCOUNT)
    return lease,job,guard,env,now

def test_actual_scheduler_clock_and_independent_guard(bound):
    lease,job,guard,env,now=bound;g.validate_job(job,lease,now);g.ensure(lease,guard,now,env)

@pytest.mark.parametrize('field,value',[('JobId','other'),('Account','other'),('JobState','PENDING'),('AllocTRES','gres/gpu=2'),('NodeList','other'),('EndTime','2026-11-06T20:00:00')])
def test_wrong_actual_scheduler_state_rejected(bound,field,value):
    lease,job,_,_,now=bound;job[field]=value
    with pytest.raises(ValueError):g.validate_job(job,lease,now)

def test_stale_guard_or_wrong_job_environment_rejected(bound):
    lease,_,guard,env,now=bound
    with pytest.raises(RuntimeError):g.ensure(lease,guard,now+46,env)
    env['SLURM_JOB_ID']='other'
    with pytest.raises(RuntimeError):g.ensure(lease,guard,now,env)

def test_native_entrypoint_dispatches_slurm_without_gcp(bound,tmp_path,monkeypatch):
    from portable.venus.native_app import ensure_guard
    lease,_,guard,env,now=bound
    for key,value in env.items():monkeypatch.setenv(key,value)
    for name,value in [('lease',lease),('guard',guard)]:
        p=tmp_path/(name+'.json');p.write_text(json.dumps(value));monkeypatch.setenv('DUPLEX_LIVE_'+name.upper()+'_FILE',str(p))
    monkeypatch.setattr(g.time,'time',lambda:now)
    ensure_guard()
