import json
from pathlib import Path
from datetime import datetime
import pytest
from training.venus import tts_completion_capacity_guard as g


def test_no_capacity_authority_means_no_provider_call(tmp_path,monkeypatch):
    from training.venus import tts_stop_controller as ctl
    monkeypatch.setattr(g,'AUTH',tmp_path/'absent.json')
    for key in ('AUTH','PLAN','KEY','R','NAME','ZONE'):monkeypatch.setattr(ctl,key,getattr(ctl,key))
    monkeypatch.setattr(ctl.c,'cloud',lambda *a,**kw:pytest.fail('No provider calls without placement authority'))
    with pytest.raises(FileNotFoundError):ctl.main(completion_capacity=True)


@pytest.mark.parametrize('zone',['us-central1-c','us-central1-b'])
def test_exact_scope_lease_and_both_approved_placements(tmp_path,monkeypatch,zone):
    from portable.venus import guard
    from training.venus.paced_guard import execution_plan
    proposal=Path('docs/NATIVE-VENUS-TTS-COMPLETION-CAPACITY-PROPOSAL-2026-11-03.json')
    auth=tmp_path/'fixture-only.json';auth.write_text(json.dumps(dict(status='APPROVED',human_approval='Approve C/B fallback within existing caps',proposal_path=str(proposal),proposal_sha256=g.sha(proposal),fixture_only=True,recorded_utc='2026-11-02T23:00:00+00:00')))
    monkeypatch.setattr(g,'AUTH',auth)
    vm=json.loads(Path('runs/venus-training-20261102/tts-placement-v56/provider-start.json').read_text())
    vm.update(id='111',name='duplex-venus-tts-20261102-'+zone[-1]+'-v59',zone=vm['zone'].replace('us-central1-f',zone))
    vm['disks'][0]['source']=vm['disks'][0]['source'].replace('us-central1-f',zone).replace('-f-v56','-'+zone[-1]+'-v59');vm['scheduling']['maxRunDuration']['seconds']='1200'
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp();storage=start-60
    plan=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=24,maximum_gpu_seconds=1200,maximum_new_running_intervals=1,vm_id='111',disk_id='222',snapshot_id='333',snapshot_source_disk_id='4375908109499109167',vm_name=vm['name'],disk_name='duplex-venus-tts-data-20261102-'+zone[-1]+'-v59',storage_started_unix=storage,storage_deadline_unix=storage+7200)
    p=tmp_path/'execution.json';p.write_text(json.dumps(plan));monkeypatch.setattr(g,'PLAN',p)
    lease=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),execution_plan_sha256=g.sha(p),vm_id='111',disk_id='222',snapshot_id='333',disk_size_gib=100,zone=zone,creation_timestamp=vm['creationTimestamp'],stop_unix=start+1200,prior_new_gpu_seconds_upper=33500,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.687,storage_started_unix=storage,storage_deadline_unix=storage+7200)
    original_sha=g.sha
    def git_deployed_sha(path):
        if str(path)=='runs/venus-training-20261102/tts-completion-v58/events.json':
            pytest.fail('Guest cannot depend on an ignored Mac-only receipt')
        return original_sha(path)
    monkeypatch.setattr(g,'sha',git_deployed_sha)
    assert execution_plan(lease)==p and guard.validate(vm,[vm],[],lease,start+1)==start+1200
    for change in (dict(zone='us-central1-f'),dict(stop_unix=start+1800),dict(prior_new_gpu_seconds_upper=0),dict(storage_deadline_unix=storage+7300),dict(disk_id='4375908109499109167')):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**change),start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm,dict(vm,id='999')],[],lease,start+1)
    plan['new_optimizer_updates_authorized_by_this_execution']=1;p.write_text(json.dumps(plan));lease['execution_plan_sha256']=g.sha(p)
    with pytest.raises(AssertionError):g.resource(lease)


def test_real_fallback_allocator_reaches_snapshot_only_after_proof(tmp_path,monkeypatch):
    from training.venus import tts_placement as placement
    proposal=Path('docs/NATIVE-VENUS-TTS-COMPLETION-CAPACITY-PROPOSAL-2026-11-03.json')
    auth=tmp_path/'fixture-only.json';auth.write_text(json.dumps(dict(status='APPROVED',proposal_path=str(proposal),proposal_sha256=g.sha(proposal))))
    monkeypatch.setattr(g,'AUTH',auth);monkeypatch.setattr(placement.c,'R',tmp_path)
    class ReachedSnapshot(Exception):pass
    def snapshot():raise ReachedSnapshot
    monkeypatch.setattr(placement,'snapshot',snapshot)
    with pytest.raises(ReachedSnapshot):placement.allocate('59')
    assert json.loads((tmp_path/'storage-deadline.json').read_text())['storage_deadline_unix']>0
