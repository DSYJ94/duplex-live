import copy
import pytest
from training.venus.resume_native import validate_state,verify_proof
from training.venus.large_inputs import PARENT_SHA,MANIFEST_SHA


def state_fixture():
    e=dict(actual_new_updates=8,source='original',checkpoint_sha256='saved8')
    r=dict(epochs=1,updates=60,train_count=240,dev_count=64,rank=256,alpha=256,last_layers=36,microbatch=1,accumulation=4,learning_rate=1e-5,warmup_updates=16,ordinary_kl_weight=.1,ordinary_hidden_weight=.01,action_aux_weight=.25,forward_path='native-segmented-bptt-v1',activation_offload_policy='repeat4-prefix-pageable-v3-layout',opal=dict(backend='keep',agent_budget_usd=0))
    s=dict(step=8,optimizer=dict(update=8),metadata=dict(source='original',arm='main',capacity_schema='native-capacity-trained-v1',parent_training_steps=466,parent_checkpoint_sha256=PARENT_SHA,cache_sha256=MANIFEST_SHA,expansion=dict(parameters=698351616),adopted=False,recipe=r))
    return s,e


def test_saved8_state_rejects_replayed_clock_and_changed_recipe():
    s,e=state_fixture();assert validate_state(s,e)['source']=='original'
    for change in [lambda x:x.update(step=0),lambda x:x['optimizer'].update(update=9),lambda x:x['metadata'].update(source='new-fake-source'),lambda x:x['metadata']['recipe'].update(learning_rate=2e-5),lambda x:x['metadata']['recipe'].update(epochs=2),lambda x:x['metadata']['recipe'].update(ordinary_kl_weight=0)]:
        broken=copy.deepcopy(s);change(broken)
        with pytest.raises(ValueError):validate_state(broken,e)


def test_independent_proof_binds_actual_boot_pid_checkpoint_and_execution():
    _,e=state_fixture();lease=dict(execution_source='newdriver',vm_id='samevm',disk_id='samedisk')
    p=dict(passed=True,actual_CUDA=True,all_Adam_OPAL_adapter_RNG_exact=True,checkpoint_sha256='saved8',saved_step=8,source='original',new_optimizer_updates=0,execution_source='newdriver',vm_id='samevm',disk_id='samedisk',boot_id='newboot',restore_pid=55)
    verify_proof(p,e,lease,77,'newboot')
    for key,value in [('actual_CUDA',False),('restore_pid',77),('boot_id','oldboot'),('execution_source','otherdriver'),('checkpoint_sha256','initial0'),('new_optimizer_updates',1),('all_Adam_OPAL_adapter_RNG_exact',False)]:
        with pytest.raises(ValueError):verify_proof(dict(p,**{key:value}),e,lease,77,'newboot')


def test_real_keep_state_after8_resumes_exact_update9_without_replay(monkeypatch,tmp_path):
    from training.venus import train
    from training.venus.native_optimizer import NativeOpal,assert_optimizer_equal
    from training.venus.lora import adapter_state
    from training.venus.state import save,load,exact_state
    from evals.harness.test_venus_native_opal import setup,make,SOURCE
    if not SOURCE.exists():pytest.skip('Pinned upstream unavailable')
    recipe=setup(monkeypatch);m,opt,rows=make();runtime=NativeOpal(opt,m,source=SOURCE,run_dir=tmp_path/'original',recipe=recipe)
    restored=None
    try:
        for i in range(8):train.train_step(m,runtime.optimizer,None,rows,i,opal=runtime)
        meta=dict(source='fixture',recipe=recipe);save(tmp_path/'saved8.pt',m,runtime.optimizer,8,meta,diagnostics_state=runtime.diagnostics_state)
        expected=train.train_step(m,runtime.optimizer,None,rows,8,opal=runtime)
        expected_state=copy.deepcopy(runtime.optimizer.state_dict());expected_weights=adapter_state(m)
        fresh,base,fr=make();restored=NativeOpal(base,fresh,source=SOURCE,run_dir=tmp_path/'fresh',recipe=recipe)
        assert load(tmp_path/'saved8.pt',fresh,restored.optimizer,meta,restored.restore_diagnostics)==8
        actual=train.train_step(fresh,restored.optimizer,None,fr,8,opal=restored)
        assert restored.optimizer.update==9
        assert_optimizer_equal(expected_state,restored.optimizer.state_dict())
        assert exact_state(expected_weights,adapter_state(fresh))
        assert actual['lr']==expected['lr'] and actual['objective']==expected['objective']
        with pytest.raises(ValueError,match='never repeat'):restored.step(None,7)
    finally:
        runtime.close()
        if restored:restored.close()
