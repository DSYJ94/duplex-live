"""Actual native train_step seam on a tiny CPU Qwen; not9B/GPU qualification."""
import copy
from pathlib import Path
import pytest
import torch
from training.venus import train
from training.venus.native_optimizer import NativeOpal,assert_optimizer_equal
from training.venus.lora import install,adapter_state,scene_loss
from training.venus.state import exact_state,save,load
from evals.harness.test_venus_training import model,example

SOURCE=Path('/private/tmp/duplex-opal-upstream-20261028')


def setup(monkeypatch):
    for name in ('reset_peak_memory_stats','synchronize'):
        monkeypatch.setattr(torch.cuda,name,lambda:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(train,'scene_loss',lambda llm,ex,device:scene_loss(llm,ex,'cpu'))
    recipe=dict(train.RECIPE,updates=24,warmup_updates=4)
    monkeypatch.setattr(train,'RECIPE',recipe)
    monkeypatch.setattr(train,'read_example',lambda cache,row:row['example'])
    return recipe


def make():
    m=model();e=example(m);install(m,2,2,last_layers=1)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=train.RECIPE['learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
    rows=[dict(id='cpu-fixture-'+str(i),family=f,example=dict(e,family=f)) for i,f in enumerate(('ordinary','quiet','delegate','cancel'))]
    return m,opt,rows


@pytest.mark.skipif(not SOURCE.exists(),reason='Exact upstream checkout is optional')
def test_actual_native_training_step_keep_matches_plain_loss_gradients_weights_and_adam(monkeypatch,tmp_path):
    recipe=setup(monkeypatch);m,opt,rows=make();plain,plainopt,plainrows=make()
    # Nonzero inherited moments represent a continuation, not just zero-state Adam.
    train.train_step(m,opt,None,rows,0);train.train_step(plain,plainopt,None,plainrows,0)
    frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad}
    opal=NativeOpal(opt,m,source=SOURCE,run_dir=tmp_path/'opal',recipe=recipe)
    clip=torch.nn.utils.clip_grad_norm_;calls=[]
    def tracked(*args,**kwargs):calls.append(1);return clip(*args,**kwargs)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',tracked)
    for step in range(6):
        before=len(calls)
        actual=train.train_step(m,opal.optimizer,None,rows,step,opal=opal)
        assert len(calls)==before+1,'Only updater clips; driver must not clip too'
        expected=train.train_step(plain,plainopt,None,plainrows,step)
        assert all(actual[k]==expected[k] for k in ('ce','kl','hidden','gradient_norm','lr'))
        assert exact_state(adapter_state(m),adapter_state(plain))
        # Scheduler adds initial_lr; Adam moments and other hyperparameters must match.
        assert exact_state(opt.state_dict()['state'],plainopt.state_dict()['state'])
        assert actual['objective']==float(torch.tensor([actual['ce']+.1*actual['kl']+.01*actual['hidden']])) or abs(actual['objective']-(actual['ce']+.1*actual['kl']+.01*actual['hidden']))<1e-6
    assert all(torch.equal(p,frozen[n]) for n,p in m.named_parameters() if n in frozen)
    opal.close()


@pytest.mark.skipif(not SOURCE.exists(),reason='Exact upstream checkout is optional')
def test_native_checkpoint_restores_full_opal_and_rejects_progress_replay(monkeypatch,tmp_path):
    recipe=setup(monkeypatch);m,opt,rows=make()
    runtime=NativeOpal(opt,m,source=SOURCE,run_dir=tmp_path/'first',recipe=recipe)
    runtime.development({'ordinary':1.2,'quiet':.8})
    first=train.train_step(m,runtime.optimizer,None,rows,0,opal=runtime)
    runtime.diagnose([first],1)
    diagnostics=copy.deepcopy(runtime.diagnostics_state)
    meta=dict(source='tiny-native-seam-proof',recipe=recipe)
    save(tmp_path/'step.pt',m,runtime.optimizer,1,meta,diagnostics_state=diagnostics)
    expected=train.train_step(m,runtime.optimizer,None,rows,1,opal=runtime)
    state=copy.deepcopy(runtime.optimizer.state_dict());weights=adapter_state(m)
    fresh,base,fr=make();restored=NativeOpal(base,fresh,source=SOURCE,run_dir=tmp_path/'restored',recipe=recipe)
    assert load(tmp_path/'step.pt',fresh,restored.optimizer,meta,restored.restore_diagnostics)==1
    assert restored.diagnostics_state==diagnostics
    actual=train.train_step(fresh,restored.optimizer,None,fr,1,opal=restored)
    assert exact_state(weights,adapter_state(fresh))
    assert_optimizer_equal(state,restored.optimizer.state_dict())
    assert all(expected[k]==actual[k] for k in ('ce','kl','hidden','gradient_norm','lr','objective'))
    with pytest.raises(ValueError,match='never repeat'):restored.step(torch.tensor(1.),0)
    runtime.close();restored.close()


def test_next_opal_continuation_recognises244_without_replaying_saved54():
    from training.venus.native_optimizer import verified_parent
    state=dict(step=54,metadata=dict(parent_training_steps=190,source='6ea80a5d2512d981d8ddf1d64c4675684667429a'))
    digest='c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270'
    assert verified_parent(state,digest,allow244=True)==244
    with pytest.raises(ValueError):verified_parent(state,digest)
    with pytest.raises(AssertionError):verified_parent(dict(state,step=53),digest,allow244=True)


def test_live_only_lease_cannot_start_native_opal_training(tmp_path):
    from types import SimpleNamespace
    lease=tmp_path/'lease.json';lease.write_text('{"resource_key":"venus-paced-recovery-v31"}')
    if not SOURCE.exists():pytest.skip('Pinned source optional')
    with pytest.raises(ValueError,match='live/evaluation-only'):
        train.main(SimpleNamespace(opal_source=SOURCE,reactive=False,lease=lease))
