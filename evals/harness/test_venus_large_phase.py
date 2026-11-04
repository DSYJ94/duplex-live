import copy
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from training.venus.large_inputs import full_native
from training.venus.lora import install, adapter_parameters, ALL_PROJECTIONS, selected_forward
from training.venus.large_capacity import expand_broad
from training.venus.capacity_state import expand_opal_state
from training.venus.large_phase import begin, write_programs
from training.venus.native_optimizer import NativeOpal, assert_optimizer_equal
from training.venus.state import rng_state, restore_rng, exact_state


def example():
    return dict(id='synthetic', fold='TRAIN', family='ordinary',
                embeddings=torch.randn(7, 32), segments=[dict(start=0, length=3), dict(start=3, length=2), dict(start=5, length=2)],
                positions=torch.tensor([2, 4]), targets=torch.tensor([7, 8]), decision_mask=torch.tensor([False, True]),
                prefix_layers=2, prefix_inputs=['must never enter broad forward'], forward_path='native-frozen-prefix-v1')


def test_all_layer_inputs_retain_exact_original_tensors_and_reject_invalid_alignment():
    original = example(); native = full_native(original)
    assert native['forward_path'] == 'native-segmented-bptt-v1'
    assert not {'prefix_layers', 'prefix_inputs'} & native.keys()
    assert original['forward_path'] == 'native-frozen-prefix-v1'
    for key in ('embeddings', 'segments', 'targets', 'positions', 'decision_mask'):
        assert native[key] is original[key]
    for change in (dict(positions=torch.tensor([1, 4])), dict(positions=torch.tensor([4, 2])),
                   dict(positions=torch.tensor([2, 6])), dict(fold='TEST'),
                   dict(segments=[dict(start=1, length=7)])):
        with pytest.raises(ValueError): full_native(dict(original, **change))


def test_native_full_path_reaches_new_earlier_layers():
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=48)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).eval(); install(model, rank=2, alpha=2, last_layers=1)
    expand_broad(model, last_layers=3, rank=8, alpha=8, seed=4)
    logits, _ = selected_forward(model, full_native(example()), 'cpu')
    logits.square().mean().backward()
    params = adapter_parameters(model)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params.values())
    assert torch.count_nonzero(params['model.layers.0.mlp.down_proj.b'].grad) > 0


def test_shape_probe_preserves_full_causal_gradients_on_tiny_cpu_only():
    from training.venus.large_memory_probe import whole_backbone_native_heads
    config=Qwen3Config(hidden_size=32,intermediate_size=64,num_hidden_layers=3,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,vocab_size=48)
    config._attn_implementation='eager'
    model=Qwen3ForCausalLM(config).eval();install(model,rank=2,alpha=2,last_layers=1)
    ex=full_native(example())
    a,_=selected_forward(model,ex,'cpu');a.square().mean().backward()
    first={n:p.grad.clone() for n,p in adapter_parameters(model).items()}
    model.zero_grad(set_to_none=True)
    b,_=whole_backbone_native_heads(model,ex,'cpu');b.square().mean().backward()
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)
    for name,p in adapter_parameters(model).items():torch.testing.assert_close(first[name],p.grad,rtol=1e-5,atol=1e-7)


def test_actual_opal_capacity_boundary_keeps_parent_program_moments_rng_and_new_zero_counters(tmp_path):
    source = Path('/private/tmp/duplex-opal-upstream-20261028')
    if not (source/'.git').exists(): pytest.skip('Pinned upstream unavailable')
    torch.manual_seed(83)
    config = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=48)
    config._attn_implementation = 'eager'
    model = Qwen3ForCausalLM(config).eval(); install(model, rank=2, alpha=2, last_layers=1)
    recipe = dict(updates=8, warmup_updates=2, learning_rate=.003, gradient_clip=1.,
                  microbatch=1, accumulation=4, adam_betas=[.9,.999], adam_eps=1e-8,
                  weight_decay=.01, ordinary_kl_weight=.1, ordinary_hidden_weight=.01, action_aux_weight=.25)
    base = torch.optim.AdamW(list(adapter_parameters(model).values()), lr=.003)
    old = NativeOpal(base, model, source=source, run_dir=tmp_path/'old', recipe=recipe)
    x = torch.tensor([[1,2,3,4]])
    loss = model(input_ids=x, labels=x).loss; loss.backward(); old.step(loss, 0); base.zero_grad(set_to_none=True)
    parent = dict(optimizer=copy.deepcopy(old.optimizer.state_dict()), rng=rng_state(),
                  metadata=dict(recipe=recipe), training_diagnostics=copy.deepcopy(old.diagnostics_state))
    old.close()
    spec = expand_broad(model, last_layers=3, rank=8, alpha=8, seed=5)
    params = adapter_parameters(model)
    mapped = expand_opal_state(parent['optimizer'], spec['before_names'], params)
    preserved = write_programs(tmp_path/'preserved', mapped['files'])
    base = torch.optim.AdamW(list(params.values()), lr=.003)
    restored = NativeOpal(base, model, source=source, run_dir=tmp_path/'expanded', recipe=recipe, initial_program=preserved)
    restored.optimizer.load_state_dict(mapped); restore_rng(parent['rng'])
    revised = dict(recipe, rank=8, alpha=8, last_layers=3, projections=list(ALL_PROJECTIONS),
                   forward_path='native-segmented-bptt-v1', learning_rate=.001)
    new, proof = begin(parent, restored, mapped, model, source=source, run_dir=tmp_path/'new',
                       program_dir=tmp_path/'program', recipe=revised, expansion=spec)
    try:
        assert proof['new_optimizer_updates'] == 0 and proof['inherited_adam_exact'] and proof['rng_exact']
        assert new.optimizer.update == 0 and new.optimizer.files == mapped['files']
        assert exact_state(mapped['user_optimizers'][0]['state'], new.base.state_dict()['state'])
        assert exact_state(parent['rng'], rng_state())
        assert len({int(v['step']) for v in new.base.state_dict()['state'].values()}) == 2
        # An actual new update must touch old and new parameters with distinct
        # counters; this is a tiny CPU fixture, never a saved native update.
        model(input_ids=x, labels=x).loss.backward(); new.step(model(input_ids=x, labels=x).loss, 0)
        counters = {int(v['step']) for v in new.base.state_dict()['state'].values()}
        assert counters == {1, 2}
    finally:
        new.close()
