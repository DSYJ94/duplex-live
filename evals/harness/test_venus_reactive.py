import json
import pytest
from training.venus.reactive import ReactiveControl, atomic


def observation(control, step=8):
    row=dict(ce=.7,kl=.02,hidden=.01,gradient_norm=2.,seconds=1.,lr=.00002,
             families=['ordinary','quiet'])
    return control.observe(step,[row]*4)


def decision(control, observed, **changes):
    value=dict(identity=control.identity,sequence=1,
               observation_sha256=observed['sha256'],not_before_step=8,
               expires_after_step=24,lr_multiplier=.5,gradient_clip=.75,
               reason='Observed gradient oscillation; test a lower rate without changing data.')
    value.update(changes)
    atomic(control.folder/'decision-001.json',value)


def test_no_agent_never_blocks_or_changes_recipe(tmp_path):
    c=ReactiveControl(tmp_path,{'source':'run-one'})
    assert not c.boundary(0) and not c.boundary(2)
    assert c.state['lr_multiplier']==1. and c.state['gradient_clip']==1.


def test_real_observation_bound_decision_applies_once_at_next_boundary(tmp_path):
    c=ReactiveControl(tmp_path,{'source':'run-one'});o=observation(c)
    decision(c,o)
    assert not c.boundary(7)
    assert c.boundary(8) and not c.boundary(9)
    assert c.state['lr_multiplier']==.5 and c.state['gradient_clip']==.75
    assert c.state['decisions'][0]['applied_after_step']==8
    assert json.loads((tmp_path/'applied-001.json').read_text())==c.state['decisions'][0]


@pytest.mark.parametrize('change',[
    {'lr_multiplier':1.01},{'lr_multiplier':float('nan')},
    {'gradient_clip':.1},{'gradient_clip':True},
    {'identity':'other-run'},{'observation_sha256':'unseen'},
    {'expires_after_step':7},{'reason':''},{'sequence':2},
])
def test_invalid_advice_does_not_mutate_state(tmp_path,change):
    c=ReactiveControl(tmp_path,{'source':'one'});o=observation(c);before=c.snapshot()
    value=dict(identity=c.identity,sequence=1,observation_sha256=o['sha256'],
               not_before_step=8,expires_after_step=24,lr_multiplier=.5,
               gradient_clip=.75,reason='Bounded numerical response')
    value.update(change)
    (tmp_path/'decision-001.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):c.boundary(8)
    assert c.snapshot()==before


def test_checkpointed_control_reproduces_decision_state_and_next_advice(tmp_path):
    ident={'source':'exact','cache':'frozen','approval':'complete'}
    c=ReactiveControl(tmp_path/'old',ident);o=observation(c);decision(c,o);c.boundary(8)
    saved=c.snapshot();fresh=ReactiveControl(tmp_path/'fresh',ident,saved)
    assert fresh.snapshot()==saved
    assert not fresh.boundary(9)
    assert fresh.state['lr_multiplier']==.5
    with pytest.raises(ValueError):ReactiveControl(tmp_path/'wrong',{'source':'new'},saved)


def test_expired_decision_never_changes_weights_or_recipe(tmp_path):
    c=ReactiveControl(tmp_path,{'source':'one'});o=observation(c);decision(c,o)
    with pytest.raises(ValueError,match='expired'):c.boundary(25)
    assert c.state['lr_multiplier']==1. and not c.state['decisions']


def test_control_state_is_in_actual_optimizer_checkpoint(tmp_path):
    import torch
    from training.venus.lora import install
    from training.venus.state import save,load
    from evals.harness.test_venus_training import model
    m=model();install(m,2,2);opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    c=ReactiveControl(tmp_path/'control',{'source':'one'});o=observation(c);decision(c,o);c.boundary(8)
    path=tmp_path/'checkpoint.pt';save(path,m,opt,8,{'recipe':'unchanged'},c.snapshot())
    fresh=model();install(fresh,2,2);newopt=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad])
    assert load(path,fresh,newopt,{'recipe':'unchanged'})==8
    state=torch.load(path,weights_only=False)
    restored=ReactiveControl(tmp_path/'restored',{'source':'one'},state['reactive_control'])
    assert restored.snapshot()==c.snapshot()


def test_agent_rate_and_clip_reach_actual_optimizer_step(tmp_path,monkeypatch):
    import torch
    from training.venus import train
    from training.venus.lora import install,scene_loss
    from evals.harness.test_venus_training import model,example
    m=model();e=example(m);install(m,2,2)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad])
    c=ReactiveControl(tmp_path,{'source':'one'});o=observation(c);decision(c,o);c.boundary(8)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(train,'read_example',lambda cache,row:e)
    monkeypatch.setattr(train,'scene_loss',lambda llm,ex,device:scene_loss(llm,ex,'cpu'))
    row=train.train_step(m,opt,None,[{'id':'one','family':'ordinary'}],8,control=c)
    assert row['lr']==train.learning_rate(8)*.5
    assert opt.param_groups[0]['lr']==row['lr'] and row['gradient_clip']==.75
    assert row['optimizer_updated'] and row['gradient_norm']>0


@pytest.mark.parametrize('version,prior,duration',[('v25',9900,4500),('v26',10500,3900),('v27',12000,900),('v28',13200,900),('v29',13700,3600),('v30',14000,3300)])
def test_prepared_lease_carries_all_prior_gpu_time_and_enforces_native_stop(version,prior,duration,monkeypatch,tmp_path):
    import hashlib
    from pathlib import Path
    from training.venus.paced_guard import AUTH,PLAN_V25,PLAN_V26,PLAN_V27,PLAN_V28,PLAN_V29,PLAN_V30,LIVE_AUTH,validate
    from portable.venus.guard import resource,stamp
    plan=PLAN_V30 if version=='v30' else PLAN_V29 if version=='v29' else PLAN_V28 if version=='v28' else PLAN_V26 if version=='v26' else PLAN_V25
    if version=='v27':
        import training.venus.paced_guard as guard
        value=json.loads(PLAN_V27.read_text());value.update(status='WITHIN_APPROVED_FOCUSED_REPAIR',focused_gpu_seconds_upper_before_launch=12000,maximum_gpu_seconds=900,previous_provider_snapshot=dict(id='6750796068068004320',status='TERMINATED',lastStartTimestamp='2026-10-28T07:00:00Z',lastStopTimestamp='2026-10-28T07:25:00Z'))
        plan=tmp_path/'live-plan.json';plan.write_text(json.dumps(value));monkeypatch.setattr(guard,'PLAN_V27',plan)
    authority=LIVE_AUTH if version in ('v29','v30') else AUTH
    p=json.loads(plan.read_text());a=json.loads(authority.read_text());r=a['resources']
    start=stamp('2026-10-28T08:00:00Z')
    test_id='5521472841931556416' if version=='v28' else '12345'
    lease=dict(resource_key='venus-paced-recovery-'+version,zone=r['zone'],disk_id=r['same_existing_disk_id'],
        disk_size_gib=100,vm_id=test_id,creation_timestamp='2026-10-28T08:00:00Z',stop_unix=start+duration,
        prior_new_gpu_seconds_upper=prior,prior_gpu_seconds=54451,prior_scope_usd=91.98448055555555,
        verified_hourly_price_usd=3.687083630136986,authorization_sha256=hashlib.sha256(authority.read_bytes()).hexdigest(),
        execution_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest())
    vm=dict(id=test_id,name=p['vm_name'],status='RUNNING',machineType='/a2-highgpu-1g',zone='/'+r['zone'],
        creationTimestamp=lease['creation_timestamp'],lastStartTimestamp='2026-10-28T08:00:00Z',
        tags={'items':['duplex-venus-train-bounded']},
        disks=[dict(boot=True,autoDelete=False,source='/disks/'+r['same_existing_disk_name'])],
        scheduling=dict(provisioningModel='FLEX_START',maxRunDuration={'seconds':str(duration)},instanceTerminationAction='STOP',automaticRestart=False))
    assert resource(lease)['ident']==test_id
    assert validate(vm,[vm],[],lease,start+10)==start+duration
    for invalid in ({**lease,'prior_new_gpu_seconds_upper':0},{**lease,'prior_gpu_seconds':0},{**lease,'stop_unix':start+duration+100}):
        with pytest.raises(AssertionError):validate(vm,[vm],[],invalid,start+10)
    with pytest.raises(AssertionError):validate(vm,[vm],[],lease,start+duration)
    with pytest.raises(AssertionError):validate(vm,[vm],[{'status':'RUNNING'}],lease,start+10)
