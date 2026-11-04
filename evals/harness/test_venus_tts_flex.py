import json
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
import pytest
from training.venus import tts_completion_flex_guard as g


def test_missing_flex_authority_blocks_real_controller(tmp_path,monkeypatch):
    from training.venus import tts_stop_controller as ctl
    monkeypatch.setattr(g,'AUTH',tmp_path/'absent.json')
    for key in ('AUTH','PLAN','KEY','R','NAME','ZONE'):monkeypatch.setattr(ctl,key,getattr(ctl,key))
    monkeypatch.setattr(ctl.c,'cloud',lambda *a,**kw:pytest.fail('No provider calls before approval'))
    with pytest.raises(FileNotFoundError):ctl.main(completion_flex=True)


@pytest.mark.parametrize("version",[62,63])
def test_exact_flex_lease_storage_budget_and_full_cohort(tmp_path,monkeypatch,version):
    from portable.venus import guard
    from training.venus.paced_guard import execution_plan
    prop=Path('docs/NATIVE-VENUS-TTS-COMPLETION-FLEX-'+('RECOVERY-' if version==63 else '')+'PROPOSAL-2026-11-04.json')
    duration=1000 if version==63 else 1200;prior=33700 if version==63 else 33500
    auth=tmp_path/'fixture-only.json';auth.write_text(json.dumps(dict(status='APPROVED',human_approval='Approve recovery within unused allowance' if version==63 else 'Approve queued Flex-start within existing caps',proposal_path=str(prop),proposal_sha256=g.sha(prop),recorded_utc='2026-11-04T13:10:00+00:00',execution_version=version)));monkeypatch.setattr(g,'RECOVERY_AUTH' if version==63 else 'AUTH',auth)
    vm=json.loads(Path('runs/venus-training-20261102/tts-placement-v56/provider-start.json').read_text());vm.update(id='111',name='duplex-venus-tts-flex-20261104-a-v'+str(version),creationTimestamp='2026-11-04T14:30:00+00:00',lastStartTimestamp='2026-11-04T14:31:00+00:00',zone=vm['zone'].replace('us-central1-f','us-central1-a'))
    disk='duplex-venus-tts-flex-data-20261104-a-v'+str(version);vm['disks'][0]['source']=vm['disks'][0]['source'].replace('us-central1-f','us-central1-a').replace('duplex-venus-tts-data-20261102-f-v56',disk);vm['scheduling'].update(provisioningModel='FLEX_START',maxRunDuration=dict(seconds=str(duration)))
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();storage=start-60
    plan=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=24,maximum_gpu_seconds=duration,execution_version=version,maximum_new_running_intervals=1,maximum_queue_seconds=3600,vm_id='111',disk_id='222',snapshot_id='4724490043241393447',snapshot_source_disk_id='4375908109499109167',vm_name=vm['name'],disk_name=disk,storage_started_unix=storage,storage_deadline_unix=storage+7200)
    p=tmp_path/'execution.json';p.write_text(json.dumps(plan));monkeypatch.setattr(g,'RECOVERY_PLAN' if version==63 else 'PLAN',p)
    lease=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),execution_plan_sha256=g.sha(p),vm_id='111',disk_id='222',snapshot_id=plan['snapshot_id'],disk_size_gib=100,zone='us-central1-a',creation_timestamp=vm['creationTimestamp'],stop_unix=start+duration,prior_new_gpu_seconds_upper=prior,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=2.034,storage_started_unix=storage,storage_deadline_unix=storage+7200)
    lease['execution_version']=version
    assert execution_plan(lease)==p and guard.validate(vm,[vm],[],lease,start+1)==start+duration
    for change in (dict(zone='us-central1-c'),dict(disk_id='4375908109499109167'),dict(stop_unix=start+1800),dict(storage_deadline_unix=storage+8000),dict(prior_new_gpu_seconds_upper=0),dict(verified_hourly_price_usd=4.01)):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**change),start+1)
    wrong=dict(vm,scheduling=dict(vm['scheduling'],provisioningModel='STANDARD'))
    with pytest.raises(AssertionError):g.validate(wrong,[wrong],[],lease,start+1)
    for field,value in [('maximum_queue_seconds',7200),('required_native_cases',12),('new_optimizer_updates_authorized_by_this_execution',1)]:
        q=dict(plan,**{field:value});p.write_text(json.dumps(q));lease['execution_plan_sha256']=g.sha(p)
        with pytest.raises(AssertionError):g.resource(lease)


def allocator_fixture(tmp_path,monkeypatch,timeout=False,version=62):
    from training.venus import tts_flex_allocation as f
    prop=Path('docs/NATIVE-VENUS-TTS-COMPLETION-FLEX-'+('RECOVERY-' if version==63 else '')+'PROPOSAL-2026-11-04.json');auth=tmp_path/'fixture-only.json';auth.write_text(json.dumps(dict(status='APPROVED',human_approval='Approve recovery within unused allowance' if version==63 else 'Approve queued Flex-start within existing caps',proposal_path=str(prop),proposal_sha256=g.sha(prop))))
    monkeypatch.setattr(f,'RECOVERY_AUTH' if version==63 else 'AUTH',auth);vm_name,disk_name,_,_=f.configuration(str(version));monkeypatch.setattr(f.c,'R',tmp_path);monkeypatch.setattr(f.c,'record',lambda *a,**kw:None)
    clock=SimpleNamespace(now=0);clock.time=lambda:clock.now
    def sleep(n):clock.now+=n
    clock.sleep=sleep;monkeypatch.setattr(f,'time',clock)
    snap=dict(id='4724490043241393447',name='owned',sourceDiskId='4375908109499109167')
    disk=dict(id='222',name=disk_name,sourceSnapshotId=snap['id'],sizeGb='100')
    state=dict(disk=False,vm=None,creates=[],runs=[],snapshot_retired=False)
    monkeypatch.setattr(f.placement,'snapshot',lambda:snap)
    monkeypatch.setattr(f.placement,'retire_snapshot',lambda s:state.update(snapshot_retired=True))
    def run(argv,seconds):
        state['runs'].append(argv)
        if argv[:4]==['gcloud','compute','disks','create']:state['disk']=True
        elif argv[:4]==['gcloud','compute','instances','delete']:state['vm']=None
        elif argv[:4]==['gcloud','compute','disks','delete']:state['disk']=False
        else:pytest.fail('Unexpected mutation '+str(argv))
        return ''
    monkeypatch.setattr(f.c,'run',run)
    def cloud(*args):
        if args[:3]==('compute','instances','list'):return [state['vm']] if state['vm'] else []
        if args[:3]==('compute','disks','list'):return [disk] if state['disk'] else []
        if args[:3]==('compute','disks','describe'):return disk
        if args[:3]==('compute','instances','describe'):return state['vm']
        if args[:3]==('compute','instances','create'):
            state['creates'].append(args);state['vm']=dict(id='111',name=vm_name,status='PENDING',disks=[dict(source='/disks/'+disk_name)])
            return dict(name='one-op',status='RUNNING')
        if args[:3]==('compute','operations','describe'):
            if timeout:return dict(name='one-op',status='RUNNING')
            state['vm'].update(status='RUNNING',lastStartTimestamp='2026-11-04T11:00:00+00:00');return dict(name='one-op',status='DONE')
        pytest.fail('Unexpected provider operation '+str(args))
    monkeypatch.setattr(f.c,'cloud',cloud)
    return f,state


@pytest.mark.parametrize("version",[62,63])
def test_one_real_flex_create_with_fixed_wait_and_native_stop(tmp_path,monkeypatch,version):
    f,state=allocator_fixture(tmp_path,monkeypatch,version=version)
    vm,disk,zone,snapshot,deadline=f.allocate(str(version))
    assert vm['status']=='RUNNING' and len(state['creates'])==1 and deadline==7200
    args=state['creates'][0]
    assert '--provisioning-model=FLEX_START' in args and '--request-valid-for-duration=3600s' in args and '--max-run-duration='+str(1000 if version==63 else 1200)+'s' in args
    assert disk['id']!='4375908109499109167' and zone=='us-central1-a'


@pytest.mark.parametrize("version",[62,63])
def test_timed_out_queue_cancels_exact_envelope_never_resubmits(tmp_path,monkeypatch,version):
    f,state=allocator_fixture(tmp_path,monkeypatch,timeout=True,version=version)
    with pytest.raises(AssertionError,match='no resubmission'):f.allocate(str(version))
    assert len(state['creates'])==1 and state['vm'] is None and not state['disk'] and state['snapshot_retired']
    assert sum(a[:4]==['gcloud','compute','instances','delete'] for a in state['runs'])==1


def test_unknown_flex_generation_cannot_borrow_authority():
    with pytest.raises(AssertionError):g.binding(dict(resource_key=g.KEY,execution_version=64))
