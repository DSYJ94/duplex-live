"""Explicit broad-adapter phase boundary after exact expanded OPAL restore.

The legacy phase_transition.begin still forbids architecture changes. Only this
separate boundary accepts a declared expansion; it never advances an optimizer.
"""
import copy
import hashlib
import importlib
from pathlib import Path

from .native_optimizer import NativeOpal, assert_optimizer_equal
from .state import exact_state, restore_rng, rng_state


def write_programs(folder, files):
    folder = Path(folder)
    if not files or any(Path(name).name != name or not name.endswith('.py') for name in files):
        raise ValueError('Exact saved OPAL Python program files required')
    folder.mkdir(parents=True, exist_ok=False)
    for name, text in files.items():
        (folder/name).write_text(text)
    return folder


def begin(parent, restored, expanded_state, model, *, source, run_dir, program_dir,
          recipe, expansion):
    from .lora import ALL_PROJECTIONS, adapter_parameters
    from .capacity_state import expand_opal_state
    if restored._closed:
        raise ValueError('Live independently restored expanded OPAL state required')
    parameters = adapter_parameters(model)
    if (expansion['format'] != 'native-broad-adapter-v1'
            or expansion['parameters'] != sum(p.numel() for p in parameters.values())
            or expansion['after_names'] != list(parameters)
            or recipe['last_layers'] != expansion['last_layers']
            or recipe['rank'] != expansion['effective_rank']
            or recipe['alpha'] != expansion['effective_alpha']
            or recipe['projections'] != list(ALL_PROJECTIONS)
            or recipe['forward_path'] != 'native-segmented-bptt-v1'):
        raise ValueError('Exact declared broad architecture and full native path required')
    # Re-derive the map, rather than accepting arbitrary replacement moments or
    # silently dropping the parent's scheduler, programs or observations.
    expected = expand_opal_state(parent['optimizer'], expansion['before_names'], parameters)
    assert_optimizer_equal(expected, expanded_state)
    assert_optimizer_equal(expected, restored.optimizer.state_dict())
    before_rng = rng_state()
    if not exact_state(before_rng, parent['rng']):
        raise ValueError('Exact parent RNG required before new schedule boundary')
    retained = ('microbatch', 'accumulation', 'adam_betas', 'adam_eps', 'weight_decay',
                'gradient_clip', 'ordinary_kl_weight', 'ordinary_hidden_weight', 'action_aux_weight')
    if any(recipe[k] != parent['metadata']['recipe'][k] for k in retained):
        raise ValueError('Capacity phase must retain complete Adam and loss recipe')
    programs = write_programs(program_dir, expected['files'])
    old_adam = copy.deepcopy(restored.base.state_dict())
    restored.close()
    runtime = NativeOpal(restored.base, model, source=source, run_dir=run_dir,
                         recipe=recipe, backend='keep', agent_budget=0., initial_program=programs)
    try:
        schedule = copy.deepcopy(runtime.scheduler.state_dict())
        with importlib.import_module('opal.programs').keep_param_group_identity():
            runtime.optimizer.updater.load_state_dict(expected['updater'])
        runtime.scheduler.load_state_dict(schedule)
        for group, rate, initial in zip(runtime.base.param_groups, schedule['_last_lr'], schedule['base_lrs']):
            group['lr'], group['initial_lr'] = rate, initial
        actual = runtime.base.state_dict()
        if not exact_state(old_adam['state'], actual['state']):
            raise ValueError('Capacity boundary changed expanded Adam moments/counters')
        for old, new in zip(old_adam['param_groups'], actual['param_groups']):
            excluded = {'lr', 'initial_lr'}
            if not exact_state({k:v for k,v in old.items() if k not in excluded},
                               {k:v for k,v in new.items() if k not in excluded}):
                raise ValueError('Undeclared Adam group change at capacity boundary')
        if runtime.optimizer.files != expected['files'] or runtime.optimizer.update != 0:
            raise ValueError('Exact inherited programs and new local clock0 required')
        runtime.diagnostics_state = dict(parent['training_diagnostics'], last_review_step=0)
        restore_rng(before_rng)
        proof = dict(new_optimizer_updates=0, inherited_adam_exact=True, parent_checkpoint_preserved=True,
                     rng_exact=exact_state(before_rng, rng_state()), phase_local_update=0,
                     expansion=expansion,
                     program_hashes={n:hashlib.sha256(t.encode()).hexdigest() for n,t in expected['files'].items()},
                     declared_changes=['all attention+MLP adapter coverage and rank',
                                       'full native segmented autograd instead of frozen35-layer cache',
                                       'new local warmup/cosine and OPAL analysis clock; original history archived'],
                     training_authorized_by_this_helper=False)
        return runtime, proof
    except BaseException:
        runtime.close()
        raise
