"""Run complete benchmark aggregation on real tiny-model gradients, no GPU proof."""
import copy
from types import SimpleNamespace
import torch
from training.venus import large_native
from training.venus.lora import install,adapter_parameters,adapter_state,scene_loss
from training.venus.state import exact_state
from evals.harness.test_venus_training import model,example


def test_complete_benchmark_uses_parameters_preserves_weights_and_marks_CPU_nonqualifying(monkeypatch,tmp_path):
    m=model();install(m,2,2);e=example(m)
    e.update(segments=[dict(start=i,length=1,return_logits=True) for i in range(6)],forward_path='native-segmented-bptt-v1')
    params=adapter_parameters(m);base=torch.optim.AdamW(list(params.values()),lr=.001)
    loss,_=scene_loss(m,e,'cpu');loss.backward();base.step();base.zero_grad(set_to_none=True)
    before=copy.deepcopy(adapter_state(m));state=copy.deepcopy(base.state_dict())
    monkeypatch.setattr(large_native,'guarded',lambda *args:None)
    rows=[dict(id=str(i),family=e['family']) for i in range(4)]
    result=large_native.benchmark(m,SimpleNamespace(base=base),SimpleNamespace(read=lambda r:e,loss=scene_loss),
                                  rows,'unused',offload=False,device='cpu',progress_path=tmp_path/'progress.jsonl')
    assert result['effective_batch']==4 and len(result['examples'])==4
    assert result['gradient_norm']>0 and result['all_Adam_moments_materialized']
    assert not result['actual_CUDA'] and result['nonqualifying_CPU_fixture']
    assert exact_state(before,adapter_state(m)) and exact_state(state,base.state_dict())
    assert len((tmp_path/'progress.jsonl').read_text().splitlines())==8
    assert all(p.grad is None for p in params.values())
