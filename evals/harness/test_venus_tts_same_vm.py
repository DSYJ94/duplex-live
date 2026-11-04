import json,copy
from pathlib import Path
from datetime import datetime,timezone
from types import SimpleNamespace
import pytest
from training.venus import tts_completion_same_vm_guard as g


def test_missing_approval_blocks_controller(tmp_path,monkeypatch):
    from training.venus import tts_stop_controller as ctl
    monkeypatch.setattr(g,'AUTH',tmp_path/'absent.json')
    for key in ('AUTH','PLAN','KEY','R','NAME','ZONE'):monkeypatch.setattr(ctl,key,getattr(ctl,key))
    monkeypatch.setattr(ctl.c,'cloud',lambda *a,**kw:pytest.fail('No provider call before authority'))
    with pytest.raises(FileNotFoundError):ctl.main(completion_same_vm=True)


def test_exact_existing_vm_budget_complete24_and_window(tmp_path,monkeypatch):
    from portable.venus import guard
    from training.venus.paced_guard import execution_plan
    proposal=Path('docs/NATIVE-VENUS-TTS-COMPLETION-SAME-VM-PROPOSAL-2026-11-04.json')
    auth=tmp_path/'fixture.json';auth.write_text(json.dumps(dict(status='APPROVED',human_approval='Approve existing-VM retry within existing caps',proposal_path=str(proposal),proposal_sha256=g.sha(proposal),recorded_utc='2025-01-01T00:00:00+00:00')));monkeypatch.setattr(g,'AUTH',auth)
    vm=json.loads(Path('runs/venus-training-20261101/source-audit-v47/provider-start.json').read_text());vm['scheduling']['maxRunDuration']['seconds']='1200'
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();prestart=start-10
    plan=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=24,maximum_gpu_seconds=1200,maximum_new_running_intervals=1,maximum_prestart_attempts=3,actual_start_attempt_index=2,prestart_started_unix=prestart,prestart_deadline_unix=prestart+2700)
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan));monkeypatch.setattr(g,'PLAN',p)
    lease=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),execution_plan_sha256=g.sha(p),vm_id=vm['id'],disk_id='4375908109499109167',disk_size_gib=100,zone='us-central1-a',creation_timestamp=vm['creationTimestamp'],stop_unix=start+1200,prior_new_gpu_seconds_upper=33500,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.687,prestart_started_unix=prestart,prestart_deadline_unix=prestart+2700)
    assert execution_plan(lease)==p and guard.validate(vm,[vm],[],lease,start+1)==start+1200
    for change in (dict(disk_id='222'),dict(zone='us-central1-c'),dict(stop_unix=start+1800),dict(prior_new_gpu_seconds_upper=0),dict(prestart_deadline_unix=prestart+2800)):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**change),start+1)
    for key,value in [('new_optimizer_updates_authorized_by_this_execution',1),('required_native_cases',12),('actual_start_attempt_index',4)]:
        q=dict(plan,**{key:value});p.write_text(json.dumps(q));lease['execution_plan_sha256']=g.sha(p)
        with pytest.raises(AssertionError):g.resource(lease)


def fixture(monkeypatch,tmp_path,outcomes):
    from training.venus import tts_same_vm_start as s
    price=json.loads(Path('runs/venus-training-20261101/current-a100-price.json').read_text())
    clock=SimpleNamespace(now=datetime.fromisoformat(price['at_utc']).timestamp()+1)
    clock.time=lambda:clock.now
    def sleep(n):clock.now+=n
    clock.sleep=sleep;monkeypatch.setattr(s,'time',clock)
    old=dict(id='3873014408897233554',name='duplex-venus-source-20261101-v46',status='TERMINATED',lastStartTimestamp='2026-11-01T00:00:00+00:00')
    actual=copy.deepcopy(old);calls=[]
    def cloud(*args):
        if args[:3]==('compute','instances','list'):return [dict(actual)]
        if args[:3]==('compute','instances','describe'):return dict(actual)
        if args[:3]==('compute','instances','start'):
            calls.append(clock.now);outcome=outcomes[len(calls)-1]
            if outcome=='timeout':raise TimeoutError('Observation timeout')
            op=dict(name='op'+str(len(calls)),status='DONE')
            if outcome in ('success','partial'):
                actual.update(status='RUNNING',lastStartTimestamp=datetime.fromtimestamp(clock.now,timezone.utc).isoformat())
            if outcome!='success':op['error']=dict(errors=[dict(code='ZONE_RESOURCE_POOL_EXHAUSTED' if outcome in ('stockout','partial') else 'PERMISSION_DENIED')])
            return op
        pytest.fail('Unexpected provider mutation '+str(args))
    monkeypatch.setattr(s.c,'cloud',cloud);monkeypatch.setattr(s.c,'record',lambda *a,**kw:None);monkeypatch.setattr(s.c,'R',tmp_path);monkeypatch.setattr(s.c,'source','fixture');monkeypatch.setattr(s,'read_pods',lambda:[])
    claim=dict(maximum_prestart_attempts=3,prestart_started_unix=clock.now,prestart_deadline_unix=clock.now+2700)
    return s,old,claim,calls,clock


def test_three_stockouts_spaced_ten_minutes_and_no_fourth(tmp_path,monkeypatch):
    s,old,claim,calls,clock=fixture(monkeypatch,tmp_path,['stockout']*3)
    with pytest.raises(RuntimeError,match='three'):s.start(old,claim)
    assert len(calls)==3 and all(b-a>=600 for a,b in zip(calls,calls[1:]))


def test_first_actual_start_ends_retry_loop(tmp_path,monkeypatch):
    s,old,claim,calls,clock=fixture(monkeypatch,tmp_path,['stockout','success'])
    vm,attempt=s.start(old,claim)
    assert vm['status']=='RUNNING' and attempt==2 and len(calls)==2 and calls[1]-calls[0]>=600


@pytest.mark.parametrize('outcome,exception',[('timeout',TimeoutError),('partial',AssertionError),('permission',AssertionError)])
def test_never_retry_timeout_partial_start_or_access_failure(tmp_path,monkeypatch,outcome,exception):
    s,old,claim,calls,clock=fixture(monkeypatch,tmp_path,[outcome])
    with pytest.raises(exception):s.start(old,claim)
    assert len(calls)==1


def test_expired_window_submits_nothing(tmp_path,monkeypatch):
    s,old,claim,calls,clock=fixture(monkeypatch,tmp_path,[]);clock.now=claim['prestart_deadline_unix']
    with pytest.raises(AssertionError):s.start(old,claim)
    assert not calls
