"""Temporary synthetic admission fixtures; never create a real approval or GPU."""
from datetime import datetime
import json

import pytest

from training.venus import british_speech_guard as g


@pytest.fixture
def admitted(tmp_path,monkeypatch):
    for key in ('AUTH','PROPOSAL','PLAN','BACKUP'):monkeypatch.setattr(g,key,tmp_path/(key.lower()+'.json'))
    proposal=dict(maximum_new_actual_GPU_intervals=3,prior_all14_GPU_upper_seconds=45730,combined_remaining_GPU_seconds_including_all_shutdown_reserves=3370,first_actual_interval_ordinal=15,first_maximum_native_GPU_seconds=3270,per_interval_shutdown_reserve_seconds=100,original_phase_all_GPU_upper_seconds_maximum=49100,original_aggregate_seconds_limit=73600,same_GPU_id=g.VM_ID,same_disk_id=g.DISK_ID,new_optimizer_updates=0,new_CPU_intervals=0,new_disks_snapshots=0,new_purchases=0,new_external_model_API_calls=0,phase_ceiling_USD=175,project_ceiling_USD=350)
    g.PROPOSAL.write_text(json.dumps(proposal))
    a=dict(status='APPROVED',human_reply_verbatim='SYNTHETIC TEST FIXTURE ONLY',proposal_path=str(g.PROPOSAL),proposal_sha256=g.sha(g.PROPOSAL),phase_ceiling_USD=175,project_ceiling_USD=350,maximum_new_GPU_intervals=3,maximum_new_GPU_seconds_including_reserve=3370,same_GPU_id=g.VM_ID,same_disk_id=g.DISK_ID,new_optimizer_updates=0)
    g.AUTH.write_text(json.dumps(a))
    g.BACKUP.write_text(json.dumps(dict(all_hashes_verified=True,all_link_targets_verified=True,files=1856,bytes=17330768218,links=1,inventory_sha256='e36199a1ad93b745281586c2fc839860f24cf982d7849b01060c6b0845c7d6c2')))
    common=dict(resource_key=g.KEY,authorization_sha256=g.sha(g.AUTH),vm_id=g.VM_ID,disk_id=g.DISK_ID,zone='europe-west4-a',disk_size_gib=250,prior_GPU_seconds_upper=45730,prior_gpu_seconds=134881,prior_scope_usd=165.7900361111111,actual_GPU_interval=15,maximum_native_GPU_seconds=3270,shutdown_reserve_seconds=100,storage_deadline_unix=g.DEADLINE,new_optimizer_updates=0)
    plan=dict(common,required_native_cases=64,required_voice_audition_cases=9,qualified=False,checkpoint_hashes=g.CHECKPOINTS,training_math_source='b39ce3ebe167b059b67fa3d131978e5da7c41ac1',selected_profile='rp-ben',reference_sha256='552adfe446367b04b29cdc0c715822fee95c45e485046ae50ff0a06b7afc61d0',runtime_file_sha256={f:g.sha(f) for f in g.RUNTIME_FILES},backup_receipt_sha256=g.sha(g.BACKUP))
    g.PLAN.write_text(json.dumps(plan))
    start=datetime.fromisoformat('2026-11-06T11:20:00+00:00').timestamp()
    lease=dict(common,execution_plan_sha256=g.sha(g.PLAN),stop_unix=start+3270,creation_timestamp='original-creation',verified_hourly_price_usd=5.9)
    vm=dict(id=g.VM_ID,name='duplex-venus-large-gpu-20261104-eua-v67',status='RUNNING',creationTimestamp='original-creation',machineType='/a2-ultragpu-1g',zone='/europe-west4-a',lastStartTimestamp='2026-11-06T11:20:00+00:00',tags=dict(items=['duplex-venus-train-bounded']),guestAccelerators=[dict(acceleratorCount=1,acceleratorType='/nvidia-a100-80gb')],disks=[dict(boot=True,autoDelete=False,diskSizeGb='250',source='/disks/duplex-venus-large-data-20261104-eua-v67')],scheduling=dict(provisioningModel='STANDARD',maxRunDuration=dict(seconds='3270'),instanceTerminationAction='STOP',automaticRestart=False))
    return lease,vm,start,plan


def test_exact_static_binding_and_actual_stop_clock(admitted):
    lease,vm,start,_=admitted
    assert g.resource(lease)['ident']==g.VM_ID
    assert g.validate(vm,[vm],[],lease,start+1)==start+3270


@pytest.mark.parametrize('field,value',[('required_native_cases',32),('required_voice_audition_cases',3),('qualified',True),('new_optimizer_updates',1),('prior_GPU_seconds_upper',0),('actual_GPU_interval',12),('selected_profile','british-george')])
def test_cohort_clock_training_and_voice_cannot_be_weakened(admitted,field,value):
    lease,vm,start,plan=admitted;plan[field]=value
    g.PLAN.write_text(json.dumps(plan));lease['execution_plan_sha256']=g.sha(g.PLAN)
    with pytest.raises(ValueError):g.resource(lease)


@pytest.mark.parametrize('fault',['overlap','late','price','extra_GPU','auto_delete','wrong_GPU'])
def test_live_provider_and_cost_fail_closed(admitted,fault):
    lease,vm,start,_=admitted;instances=[vm];pods=[];now=start+1
    if fault=='overlap':pods=[dict(status='RUNNING')]
    if fault=='late':now=start+3270
    if fault=='price':lease['verified_hourly_price_usd']=6
    if fault=='extra_GPU':instances.append(dict(id='other',status='RUNNING',guestAccelerators=[{}]))
    if fault=='auto_delete':vm['disks'][0]['autoDelete']=True
    if fault=='wrong_GPU':vm['id']='other'
    with pytest.raises(ValueError):g.validate(vm,instances,pods,lease,now)


def test_partial_backup_cannot_authorize_model_load(admitted):
    lease,vm,start,plan=admitted;proof=json.loads(g.BACKUP.read_text());proof['files']=164
    g.BACKUP.write_text(json.dumps(proof));plan['backup_receipt_sha256']=g.sha(g.BACKUP)
    g.PLAN.write_text(json.dumps(plan));lease['execution_plan_sha256']=g.sha(g.PLAN)
    with pytest.raises(ValueError,match='Full1856'):g.resource(lease)


def test_missing_real_approval_blocks_before_model_import(tmp_path,monkeypatch):
    from portable.venus import native_app
    monkeypatch.setattr(native_app,'ensure_guard',lambda:None)
    monkeypatch.setattr(g,'AUTH',tmp_path/'missing-real-owner-approval.json')
    path=tmp_path/'lease.json';path.write_text(json.dumps(dict(resource_key=g.KEY)))
    monkeypatch.setenv('DUPLEX_LIVE_LEASE_FILE',str(path))
    with pytest.raises(FileNotFoundError):native_app.NativeEngine('never-load',speech_completion=True,nonspoken_actions=True)


def test_shared_provider_dispatch_uses_exact_new_phase(admitted):
    from portable.venus.guard import resource,validate
    lease,vm,start,_=admitted
    assert resource(lease)['ident']==g.VM_ID
    assert validate(vm,[vm],[],lease,start+1)==start+3270


@pytest.mark.parametrize('path,digest',[(None,None),('default.wav','unknown')])
def test_composition_cannot_silently_use_default_or_unapproved_voice(admitted,tmp_path,monkeypatch,path,digest):
    from portable.venus.speech_composition import require_authorization
    lease,vm,start,_=admitted
    p=tmp_path/'lease.json';p.write_text(json.dumps(lease));monkeypatch.setenv('DUPLEX_LIVE_LEASE_FILE',str(p))
    with pytest.raises(ValueError,match='no default reference fallback'):require_authorization(path,digest)


def test_new_collector_retains_original_training_source_separately(admitted,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from training.venus import british_behavior as b
    lease,vm,start,plan=admitted
    plan['execution_source']='new-git-source';g.PLAN.write_text(json.dumps(plan))
    lease['execution_plan_sha256']=g.sha(g.PLAN)
    path=tmp_path/'lease.json';path.write_text(json.dumps(lease))
    monkeypatch.setattr(b.subprocess,'check_output',lambda *a,**kw:'new-git-source\n')
    args=SimpleNamespace(lease=path,source='old-training-source',voice_reference='not-read',voice_reference_sha256=plan['reference_sha256'])
    with pytest.raises(ValueError,match='Actual Git producer'):b.admit(args)
    args.source='new-git-source';args.voice_reference_sha256='wrong'
    with pytest.raises(ValueError,match='RP native reference'):b.admit(args)
