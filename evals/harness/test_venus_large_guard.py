import copy
import json
from datetime import datetime, timezone

import pytest

from training.venus import large_guard as g
from training.venus.large_inputs import sha, PARENT_SHA, MANIFEST_SHA


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(g, 'ROOT', tmp_path)
    now = datetime.now(timezone.utc).timestamp()
    creation = datetime.fromtimestamp(now-5, timezone.utc).isoformat()
    p = dict(resource_key=g.KEY, authorization_sha256=sha(g.AUTH),
             parent_sha256=PARENT_SHA, cache_manifest_sha256=MANIFEST_SHA,
             maximum_all_branch_updates=984, external_model_API_calls=0,
             prior_project_USD=g.OLD_USD, vm_id='123', disk_id='456', zone='us-central1-a',
             creation_timestamp=creation, storage_started_unix=now-10,
             storage_deadline_unix=now-10+86400, previous_intervals=[],
             maximum_running_seconds=43100, disk_size_gib=250,
             machine='a2-highgpu-1g', provisioning='FLEX_START', optimizer_work_started=False,
             vm_name='duplex-venus-large-gpu-v65', disk_name='duplex-venus-large-data-v65')
    path = tmp_path/'interval-01.json'; path.write_text(json.dumps(p))
    lease = dict(p, execution_plan_path=str(path), execution_plan_sha256=sha(path),
                 prior_capacity_GPU_seconds_upper=0, verified_hourly_price_usd=2.1, stop_unix=now+43100)
    vm = dict(id='123', name=p['vm_name'], status='RUNNING', creationTimestamp=creation,
              lastStartTimestamp=datetime.fromtimestamp(now,timezone.utc).isoformat(),
              machineType='machines/a2-highgpu-1g', zone='zones/us-central1-a',
              tags=dict(items=['duplex-venus-train-bounded']),
              disks=[dict(boot=True, autoDelete=False, diskSizeGb=250, source='/disks/'+p['disk_name'])],
              scheduling=dict(provisioningModel='FLEX_START', maxRunDuration=dict(seconds=43100),
                              instanceTerminationAction='STOP', automaticRestart=False))
    return p,path,lease,vm,now


def test_new_guard_dispatch_and_complete_scope(tmp_path,monkeypatch):
    p,path,lease,vm,now=fixture(tmp_path,monkeypatch)
    from portable.venus.guard import resource,validate
    assert resource(lease)['machine']=='a2-highgpu-1g'
    assert validate(vm,[vm],[dict(status='EXITED')],lease,now)==lease['stop_unix']
    for change in (dict(verified_hourly_price_usd=8.1),dict(stop_unix=lease['stop_unix']+1),
                   dict(storage_deadline_unix=lease['storage_deadline_unix']+1)):
        with pytest.raises(ValueError):validate(vm,[vm],[],dict(lease,**change),now)
    with pytest.raises(ValueError):validate(vm,[vm,dict(vm,id='999')],[],lease,now)
    with pytest.raises(ValueError):validate(vm,[vm],[dict(status='RUNNING')],lease,now)
    broken=copy.deepcopy(vm);broken['disks'][0]['autoDelete']=True
    with pytest.raises(ValueError):validate(broken,[vm],[],lease,now)


def test_aggregate_time_and_plan_hash_cannot_reset(tmp_path,monkeypatch):
    p,path,lease,vm,now=fixture(tmp_path,monkeypatch)
    p['maximum_running_seconds']=43200;path.write_text(json.dumps(p))
    with pytest.raises(ValueError):g.resource(lease)
    lease['execution_plan_sha256']=sha(path)
    with pytest.raises(ValueError,match='Aggregate'):g.resource(lease)
    p['maximum_running_seconds']=43100;p['previous_intervals']=[dict()];path.write_text(json.dumps(p))
    lease['execution_plan_sha256']=sha(path)
    with pytest.raises(ValueError,match='prior actual intervals'):g.resource(lease)


def test_80GB_fallback_requires_exact_included_scratch_and_standard_zone(tmp_path,monkeypatch):
    p,path,lease,vm,now=fixture(tmp_path,monkeypatch)
    p.update(machine='a2-ultragpu-1g',provisioning='STANDARD',zone='europe-west4-a')
    path.write_text(json.dumps(p));lease.update(zone=p['zone'],execution_plan_sha256=sha(path))
    vm.update(machineType='machines/a2-ultragpu-1g',zone='zones/europe-west4-a')
    vm['scheduling']['provisioningModel']='STANDARD'
    with pytest.raises(ValueError,match='scratch'):g.validate(vm,[vm],[],lease,now)
    vm['disks'].append(dict(type='SCRATCH',diskSizeGb=375))
    assert g.validate(vm,[vm],[],lease,now)==lease['stop_unix']
    vm['disks'].append(dict(type='SCRATCH',diskSizeGb=375))
    with pytest.raises(ValueError,match='scratch'):g.validate(vm,[vm],[],lease,now)


@pytest.mark.parametrize('number',[4,5,6,7,8])
def test_fourth_interval_requires_exact_amendment_same_resources_and_all_prior_time(tmp_path,monkeypatch,number):
    fifth=number==5;sixth=number==6
    actual_root=g.ROOT
    p,path,lease,vm,now=fixture(tmp_path,monkeypatch)
    previous=[]
    for i,upper in ([(1,300),(2,400),(3,900)]+([(4,500)] if number>=5 else [])+([(5,500)] if number>=6 else [])+([(6,100)] if number>=7 else [])+([(7,1300)] if number>=8 else [])):
        original=actual_root/('gpu-05-capacity-01' if i==5 else f'gpu-{i:02d}')/'provider-final.json'
        terminal=json.loads(original.read_text());receipt=tmp_path/f'provider-final-{i:02d}.json'
        receipt.write_bytes(original.read_bytes())
        previous.append(dict(provider_final_path=str(receipt),sha256=sha(receipt),
                             vm_id=terminal['id'],disk_id='preserved',upper_seconds=upper,optimizer_updates=0))
    p.update(vm_id='7521667153519853425',disk_id='2297341858915206463',zone='europe-west4-a',
             machine='a2-ultragpu-1g',provisioning='STANDARD',maximum_running_seconds=41500,
             vm_name='duplex-venus-large-gpu-20261104-eua-v67',disk_name='duplex-venus-large-data-20261104-eua-v67',
             storage_started_unix=1793810863.296897,storage_deadline_unix=1793897263.296897,
             memory_authorization_sha256=sha(g.MEMORY_AUTH),previous_intervals=previous)
    if fifth:p.update(prefix_authorization_sha256=sha(g.PREFIX_AUTH),maximum_running_seconds=41000,activation_memory_policy='repeat4-prefix-pageable-v2')
    if sixth:p.update(layout_authorization_sha256=sha(g.LAYOUT_AUTH),maximum_running_seconds=40500,activation_memory_policy='repeat4-prefix-pageable-v3-layout')
    if number>=7:p.update(aggregate_authorization_sha256=sha(g.AGGREGATE_AUTH),maximum_running_seconds=40400,activation_memory_policy='repeat4-prefix-pageable-v3-layout')
    if number==8:p['maximum_running_seconds']=39100
    path=tmp_path/('interval-%02d.json'%number);path.write_text(json.dumps(p))
    lease.update(p,execution_plan_path=str(path),execution_plan_sha256=sha(path),prior_capacity_GPU_seconds_upper=4000 if number==8 else 2700 if number>=7 else 2600 if sixth else 2100 if fifth else 1600)
    assert g.resource(lease)['duration']==(39100 if number==8 else 40400 if number>=7 else 40500 if sixth else 41000 if fifth else 41500)
    for key,value in [('disk_id','wrong'),('maximum_running_seconds',41600),('aggregate_authorization_sha256' if number>=7 else 'layout_authorization_sha256' if sixth else 'prefix_authorization_sha256' if fifth else 'memory_authorization_sha256','wrong'),
                      ('storage_deadline_unix',1793897264.296897)]:
        changed=dict(p,**{key:value});path.write_text(json.dumps(changed))
        bad=dict(lease,**{key:value},execution_plan_sha256=sha(path))
        with pytest.raises(ValueError):g.resource(bad)
    path.write_text(json.dumps(p));lease['execution_plan_sha256']=sha(path)
    with pytest.raises(ValueError):g.resource(dict(lease,prior_capacity_GPU_seconds_upper=0))
    monkeypatch.setattr(g,'MEMORY_AUTH',tmp_path/'missing-amendment.json')
    with pytest.raises(FileNotFoundError):g.resource(lease)
