import json
from types import SimpleNamespace

import pytest

from training.venus.large_train import admission, batches


def rows():
    return [dict(id=f'train-{i:03d}',fold='TRAIN') for i in range(240)] + [dict(id=f'dev-{i:03d}',fold='DEV') for i in range(64)]


def test_joint_admission_requires_complete_pair_and_backup_room():
    result=admission(10,20,20000,1_000_000_000)
    assert result['epochs']==8 and result['updates_each_arm']==480
    assert result['joint_required_seconds']<20000
    assert result['selection_uses_losses_or_heldout'] is False
    smaller=admission(10,20,20000,20_000_000_000)
    assert smaller['epochs']==2 and smaller['joint_backup_bytes_upper']<100_000_000_000
    for values in ((1000,100,10000,1),(1,1,10000,40_000_000_000),(0,1,10000,1)):
        with pytest.raises((ValueError,TimeoutError)):admission(*values)


def test_full_epoch_order_never_includes_dev_or_drops_train():
    actual=batches(rows(),3,20261104)
    assert len(actual)==180 and all(len(batch)==4 for batch in actual)
    expected={f'train-{i:03d}' for i in range(240)}
    for i in range(3):assert {r['id'] for batch in actual[i*60:(i+1)*60] for r in batch}==expected
    assert actual==batches(rows(),3,20261104) and actual[:60]!=actual[60:120]
    with pytest.raises(ValueError):batches(rows()[1:],1,2)


def test_existing_stage_claim_blocks_before_any_provider_call(monkeypatch,tmp_path):
    from training.venus import large_stage as stage
    monkeypatch.setattr(stage,'R',tmp_path)
    monkeypatch.setattr(stage.c,'cloud',lambda *args:pytest.fail('Duplicate provider call'))
    with pytest.raises(ValueError,match='never restart'):stage.main()


def test_disk_create_uses_supported_single_request_not_async(monkeypatch):
    from training.venus import large_stage as stage
    calls=[]
    def run(argv,timeout):
        calls.append(argv)
        assert argv[:4]==['gcloud','compute','disks','create'] and '--async' not in argv
        assert '--size=250GB' in argv and timeout==900
        return '[{"name":"verified-disk","status":"READY"}]'
    monkeypatch.setattr(stage.c,'run',run)
    assert stage.create_disk(dict(name='existing-owned-snapshot'))[0]['status']=='READY'
    assert len(calls)==1


def test_unadmitted_native_entrypoint_never_loads_weights(monkeypatch,tmp_path):
    from training.venus import large_native as native
    from training.venus import native_cache
    monkeypatch.setattr(native,'guarded',lambda *args:(_ for _ in ()).throw(ValueError('No actual resource proof')))
    monkeypatch.setattr(native_cache,'load_native',lambda *args,**kwargs:pytest.fail('Model allocated before guard'))
    output=tmp_path/'out'
    with pytest.raises(ValueError,match='actual resource proof'):
        native.prepare(SimpleNamespace(lease='missing',output=output))
    assert not output.exists()


def test_oom_fallback_exits_exception_context_and_releases_failed_graph():
    import sys,gc,weakref,torch
    from training.venus.large_native import memory_fallback
    class Graph:pass
    seen=[];refs=[]
    def attempt(offload):
        seen.append(offload)
        if not offload:
            graph=Graph();refs.append(weakref.ref(graph))
            raise torch.cuda.OutOfMemoryError('Synthetic CPU-only failure')
        assert sys.exc_info()==(None,None,None)
        assert refs[0]() is None,'Failed graph still live during retry'
        return 'measured'
    records=[]
    assert memory_fallback(attempt,gc.collect,lambda:records.append('first failure'))=='measured'
    assert seen==[False,True] and records==['first failure']


def test_explicit_new_training_hooks_preserve_native_opal_step(monkeypatch,tmp_path):
    from training.venus import train
    from training.venus.lora import scene_loss
    from training.venus.native_optimizer import NativeOpal
    from evals.harness.test_venus_native_opal import setup,make,SOURCE
    if not SOURCE.exists():pytest.skip('Pinned upstream unavailable')
    recipe=setup(monkeypatch);model,opt,selected=make()
    runtime=NativeOpal(opt,model,source=SOURCE,run_dir=tmp_path/'opal',recipe=recipe)
    seen=[]
    def reader(cache,row):seen.append(row['id']);return row['example']
    try:
        metrics=train.train_step(model,runtime.optimizer,None,selected,0,opal=runtime,
                reader=reader,loss_fn=lambda m,e,d:scene_loss(m,e,'cpu'),recipe=recipe)
        assert runtime.optimizer.update==1 and metrics['optimizer_updated']
        assert seen==[row['id'] for row in selected]
    finally:runtime.close()
