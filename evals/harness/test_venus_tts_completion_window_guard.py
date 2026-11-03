import copy,json
from pathlib import Path
import pytest
from training.venus import tts_completion_window_guard as g


def test_missing_authority_blocks_real_controller_before_provider_mutations(tmp_path,monkeypatch):
    from training.venus import tts_stop_controller as ctl
    monkeypatch.setattr(g,'AUTH',tmp_path/'not-approved.json')
    # Restore the shared controller globals after this real-entrypoint test.
    for key in ('AUTH','PLAN','KEY','R','NAME','ZONE'):monkeypatch.setattr(ctl,key,getattr(ctl,key))
    monkeypatch.setattr(ctl.c,'cloud',lambda *a,**kw:pytest.fail('No provider call is allowed before authority'))
    with pytest.raises(FileNotFoundError):ctl.main(completion_window=True)


def test_complete24_lease_exact_time_resource_and_carried_spend(tmp_path,monkeypatch):
    from portable.venus import guard
    from training.venus.paced_guard import execution_plan
    auth=tmp_path/'fixture-only-authority.json'
    proposal=Path('docs/NATIVE-VENUS-TTS-COMPLETION-WINDOW-PROPOSAL-2026-11-03.json')
    auth.write_text(json.dumps(dict(status='APPROVED',human_approval='Approve refreshed 20-minute comparison within existing caps',proposal_path=str(proposal),proposal_sha256=g.sha(proposal),fixture_only=True,recorded_utc='2026-11-02T23:00:00+00:00')))
    monkeypatch.setattr(g,'AUTH',auth)
    vm=json.loads(Path('runs/venus-training-20261102/tts-placement-v56/provider-start.json').read_text())
    vm.update(id='111',name='duplex-venus-tts-20261102-f-v58')
    vm['disks'][0]['source']=vm['disks'][0]['source'].replace('-v56','-v58');vm['scheduling']['maxRunDuration']['seconds']='1200'
    plan=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),new_optimizer_updates_authorized_by_this_execution=0,required_native_cases=24,maximum_gpu_seconds=1200,maximum_new_running_intervals=1,vm_id='111',disk_id='222',snapshot_id='333',snapshot_source_disk_id='4375908109499109167',vm_name=vm['name'],disk_name='duplex-venus-tts-data-20261102-f-v58')
    from datetime import datetime
    storage_start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()-60
    plan.update(storage_started_unix=storage_start,storage_deadline_unix=storage_start+7200)
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan));monkeypatch.setattr(g,'PLAN',p)
    from datetime import datetime
    start=datetime.fromisoformat(vm['lastStartTimestamp']).timestamp()
    lease=dict(resource_key=g.KEY,authorization_sha256=g.sha(auth),execution_plan_sha256=g.sha(p),vm_id='111',disk_id='222',snapshot_id='333',disk_size_gib=100,zone='us-central1-f',creation_timestamp=vm['creationTimestamp'],stop_unix=start+1200,prior_new_gpu_seconds_upper=33500,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,verified_hourly_price_usd=3.687,storage_started_unix=storage_start,storage_deadline_unix=storage_start+7200)
    assert execution_plan(lease)==p and guard.validate(vm,[vm],[],lease,start+1)==start+1200
    for change in (dict(stop_unix=start+1800),dict(prior_new_gpu_seconds_upper=33100),dict(zone='us-central1-c'),dict(disk_id='4375908109499109167'),dict(snapshot_id='444'),dict(storage_deadline_unix=storage_start+7300),dict(storage_started_unix=storage_start+1)):
        with pytest.raises(AssertionError):g.validate(vm,[vm],[],dict(lease,**change),start+1)
    with pytest.raises(AssertionError):g.validate(vm,[vm,dict(vm,id='999')],[],lease,start+1)
    for field,value in [('required_native_cases',12),('new_optimizer_updates_authorized_by_this_execution',1),('maximum_new_running_intervals',2)]:
        q=dict(plan,**{field:value});p.write_text(json.dumps(q));lease['execution_plan_sha256']=g.sha(p)
        with pytest.raises(AssertionError):g.resource(lease)


def test_expired_v57_window_cannot_recover_snapshot_before_failure(tmp_path,monkeypatch):
    from training.venus import tts_placement as p
    root=tmp_path/'tts-completion-v57';root.mkdir()
    old=tmp_path/'tts-placement-v56';old.mkdir();(old/'storage-deadline.json').write_text(json.dumps(dict(storage_deadline_unix=0)))
    monkeypatch.setattr(p.c,'R',root)
    monkeypatch.setattr(p,'snapshot',lambda:pytest.fail('Expired window must fail BEFORE provider mutation'))
    with pytest.raises(AssertionError,match='Expired'):p.allocate('57')
    assert not (root/'storage-deadline.json').exists()


def test_real_v58_allocator_reaches_snapshot_after_storage_proof(tmp_path,monkeypatch):
    from training.venus import tts_placement as p
    proposal=Path('docs/NATIVE-VENUS-TTS-COMPLETION-WINDOW-PROPOSAL-2026-11-03.json')
    auth=tmp_path/'fixture-only.json'
    auth.write_text(json.dumps(dict(status='APPROVED',proposal_path=str(proposal),proposal_sha256=g.sha(proposal))))
    monkeypatch.setattr(g,'AUTH',auth)
    monkeypatch.setattr(p.c,'R',tmp_path)
    class ReachedSnapshot(Exception):pass
    def snapshot():raise ReachedSnapshot
    monkeypatch.setattr(p,'snapshot',snapshot)
    with pytest.raises(ReachedSnapshot):p.allocate('58')
    assert json.loads((tmp_path/'storage-deadline.json').read_text())['storage_deadline_unix']>0


def test_resume_only_exact_terminal_import_failure_without_storage(tmp_path,monkeypatch):
    from training.venus import tts_stop_controller as ctl
    from types import SimpleNamespace
    for n,v in [('claim.json',dict(pid=123,source='failed')),('events.json',[dict(event='tts_stop_failed',error_type='NameError',error="name 'Path' is not defined")])]:
        (tmp_path/n).write_text(json.dumps(v))
    (tmp_path/'source.bundle').write_bytes(b'preserve failed source')
    (tmp_path/'bootstrap.sh').write_text('unused')
    monkeypatch.setattr(ctl.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1,stdout=''))
    monkeypatch.setattr(ctl.c,'events',[])
    (tmp_path/'storage-deadline.json').write_text('{}')
    with pytest.raises(AssertionError):ctl.resume_import_failure(tmp_path)
    (tmp_path/'storage-deadline.json').unlink()
    ctl.resume_import_failure(tmp_path)
    assert (tmp_path/'claim.json').exists() and (tmp_path/'failed-import-source.bundle').read_bytes()==b'preserve failed source'
    with pytest.raises(AssertionError):ctl.resume_import_failure(tmp_path)
