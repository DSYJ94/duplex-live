"""Sustained authority, original-proof reuse and exact one-epoch admission."""
import json
from pathlib import Path
import pytest
from training.venus import large_guard as g,sustained
from training.venus.large_inputs import sha
from evals.harness.test_venus_large_guard import fixture


def test_sustained_exact_time_budget_and_resource_binding(tmp_path,monkeypatch):
    actual=g.ROOT;p,path,lease,vm,now=fixture(tmp_path,monkeypatch);previous=[]
    for n,upper in enumerate([300,400,900,500,500,100,1300,300,4400],1):
        src=actual/('gpu-05-capacity-01' if n==5 else f'gpu-{n:02d}')/'provider-final.json'
        target=tmp_path/f'provider-final-{n:02d}.json';target.write_bytes(src.read_bytes());v=json.loads(target.read_text())
        previous.append(dict(provider_final_path=str(target),sha256=sha(target),vm_id=v['id'],disk_id='original',upper_seconds=upper,optimizer_updates=0))
    p.update(vm_id='7521667153519853425',disk_id='2297341858915206463',zone='europe-west4-a',machine='a2-ultragpu-1g',provisioning='STANDARD',vm_name='duplex-venus-large-gpu-20261104-eua-v67',disk_name='duplex-venus-large-data-20261104-eua-v67',maximum_running_seconds=64800,previous_intervals=previous,maximum_all_branch_updates=120,storage_deadline_unix=1794052800.,sustained_authorization_sha256=sha(sustained.AUTH),activation_memory_policy='repeat4-prefix-pageable-v3-layout')
    path=tmp_path/'interval-10.json'
    def bound(value):
        path.write_text(json.dumps(value));return dict(lease,**value,execution_plan_path=str(path),execution_plan_sha256=sha(path),prior_capacity_GPU_seconds_upper=8700)
    assert g.resource(bound(p))['duration']==64800
    for key,value in [('maximum_running_seconds',64801),('storage_deadline_unix',1794052801.),('sustained_authorization_sha256','wrong'),('disk_id','wrong'),('maximum_all_branch_updates',121),('optimizer_work_started',True)]:
        with pytest.raises((ValueError,KeyError)):g.resource(bound(dict(p,**{key:value})))
    good=bound(p)
    with pytest.raises(ValueError):g.resource(dict(good,prior_capacity_GPU_seconds_upper=0))
    vm.update(id=p['vm_id'],name=p['vm_name'],machineType='/a2-ultragpu-1g',zone='/europe-west4-a')
    vm['disks']=[dict(type='PERSISTENT',boot=True,autoDelete=False,diskSizeGb=250,source='/disks/'+p['disk_name']),dict(type='SCRATCH',diskSizeGb=375)]
    vm['scheduling'].update(provisioningModel='STANDARD',maxRunDuration=dict(seconds=64800))
    good.update(verified_hourly_price_usd=5.618589712328767,stop_unix=now+64800)
    assert g.validate(vm,[vm],[],good,now)==good['stop_unix']
    with pytest.raises(ValueError,match='budget'):g.validate(vm,[vm],[],dict(good,verified_hourly_price_usd=6),now)
    later=tmp_path/'interval-11.json';later.write_text(json.dumps(p))
    with pytest.raises(ValueError):g.resource(dict(good,execution_plan_path=str(later),execution_plan_sha256=sha(later)))


def test_stage_and_validate_real_original_reports_cpu_only(tmp_path,monkeypatch):
    original=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v73')
    monkeypatch.setattr(sustained,'ORIGINAL',original)
    # Avoid rehashing8GB on the Mac for a source regression; actual GPU stage
    # and fresh restore independently hash the real checkpoint without this seam.
    actual_sha=sustained.sha
    monkeypatch.setattr(sustained,'sha',lambda p:sustained.authority()['init_sha256'] if Path(p)==original/'expanded-init.pt' else actual_sha(p))
    out=tmp_path/'reuse';sustained.stage(out,'current-test-source')
    assert sustained.validate(out,'current-test-source')['original_source']=='a833a32f30aa9f99b47a15dad57b7df33f816ff6'
    assert json.loads((out/'admission.json').read_text())['source']!='current-test-source'
    assert not (out/'fresh-process-restore.json').exists()
    assert (out/'expanded-init.pt').is_symlink()
    with pytest.raises(ValueError):sustained.validate(out,'different-source')
    (out/'admission.json').write_text('{}')
    with pytest.raises(ValueError):sustained.validate(out,'current-test-source')


def test_measured_schedule_fits_one_epoch_with_approved_extension():
    from training.venus.large_train import admission
    root=Path('/Users/daryl/Documents/duplex-native-recovery-20261102/large-capacity-v73')
    old=json.loads((root/'admission.json').read_text());c=json.loads((root/'control-runtime.json').read_text())
    w=json.loads((root/'workload-geometry.json').read_text());w.pop('source')
    result=admission(old['measured']['seconds'],1957.480212193,63000,8386227230,workload=w,control=c['measured'])
    assert result['epochs']==1 and result['updates_each_arm']==60 and result['complete_train_per_epoch']==240
