import copy
from types import SimpleNamespace
import torch
import pytest
from evals.harness.test_venus_training import model,example
from training.venus.lora import install,LoRALinear
from training.venus.conditioning_probe import post_token_example,compare_fixed_history,diagnostic_rows


def fixture():
    m=model();e=example(m);e.update(id='tiny-fixture',fold='TRAIN',segments=[dict(start=i,length=1) for i in range(len(e['embeddings']))])
    install(m,2,2,last_layers=1)
    tts=SimpleNamespace(emb_text=torch.nn.Embedding(64,16),projector_semantic=torch.nn.Linear(32,16))
    return SimpleNamespace(llm=m,tts=tts),e


def test_post_token_alignment_and_condition_identity_then_adapter_drift():
    m,e=fixture();post,ids=post_token_example(m.llm,e,'cpu')
    assert post['positions'].tolist()==[4] and ids.tolist()==[5]
    result=compare_fixed_history(m,e,'cpu')
    assert result['condition_mse']==0 and result['post_token_states']
    with torch.no_grad():
        for module in m.llm.modules():
            if isinstance(module,LoRALinear):module.b.add_(.4)
    changed=compare_fixed_history(m,e,'cpu')
    assert changed['condition_mse']>0 and changed['hidden_mse']>0
    assert not changed['audible_quality_measured'] and changed['optimizer_updates']==0


def test_alignment_failure_and_heldout_data_are_rejected():
    m,e=fixture();bad=copy.deepcopy(e);bad['embeddings'][4]+=1
    with pytest.raises(ValueError,match='exact fixed'):post_token_example(m.llm,bad,'cpu')
    bad=dict(e,fold='HOLDOUT')
    with pytest.raises(ValueError,match='held-out'):post_token_example(m.llm,bad,'cpu')


def test_all_families_both_nonheldout_folds_selected_before_metrics():
    rows=[dict(id=f+'-'+family,fold=f,family=family) for f in ('TRAIN','DEV','HOLDOUT') for family in ('ordinary','delegate','quiet','cancel','pause','pending')]
    picked=diagnostic_rows({'scenes':rows})
    assert len(picked)==12 and all(r['fold']!='HOLDOUT' for r in picked)
    with pytest.raises(ValueError,match='Missing'):diagnostic_rows({'scenes':rows[:3]})


def test_probe_refuses_to_load_another_model_beside_live_gpu(monkeypatch):
    import subprocess
    from training.venus.conditioning_probe import main
    from portable.venus import native_app
    from training.venus import train
    monkeypatch.setattr(native_app,'ensure_guard',lambda:None)
    monkeypatch.setattr(train,'assert_deadline',lambda *args:None)
    monkeypatch.setattr(subprocess,'check_output',lambda *args,**kwargs:'999999\n')
    with pytest.raises(RuntimeError,match='Another GPU process'):
        main(SimpleNamespace(lease='not-read'))
