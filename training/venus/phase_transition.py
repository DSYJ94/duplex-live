"""Explicit new phase after exact wrapped-parent restore, without resetting Adam.

Parent clocks and analysis history remain in the immutable checkpoint. The new
phase has a declared local clock, inherits executable programs and updater state,
and starts its separately admitted schedule. This module authorizes no updates.
"""
import copy
import hashlib
import importlib
from pathlib import Path

from .native_optimizer import NativeOpal, assert_optimizer_equal
from .state import exact_state, rng_state, restore_rng


def begin(parent, restored, model, *, source, run_dir, program_dir, recipe,
          backend='keep', agent_budget=0.):
    expected = parent['optimizer']
    assert_optimizer_equal(expected, restored.optimizer.state_dict())
    if restored._closed:
        raise ValueError('Verify the live restored parent before a phase transition')
    before_rng = rng_state()
    if not exact_state(before_rng, parent['rng']):
        raise ValueError('Restore original RNG before the explicit phase boundary')
    old_recipe = parent['metadata']['recipe']
    retained = ('rank','alpha','last_layers','projections','microbatch','accumulation',
                'adam_betas','adam_eps','weight_decay','gradient_clip',
                'ordinary_kl_weight','ordinary_hidden_weight','forward_path')
    if any(recipe[key] != old_recipe[key] for key in retained):
        raise ValueError('This phase preserves architecture, Adam and retention recipe')
    if len(expected['user_optimizers']) != 1 or len(expected['user_schedulers']) != 1:
        raise ValueError('Exact one-Adam/one-scheduler native lineage required')
    program_dir, run_dir = Path(program_dir), Path(run_dir)
    if program_dir.exists() or run_dir.exists():
        raise ValueError('Fresh phase/program directories required')
    program_dir.mkdir(parents=True)
    for name, text in expected['files'].items():
        if Path(name).name != name or not name.endswith('.py'):
            raise ValueError('Only preserved native OPAL program files allowed')
        (program_dir/name).write_text(text)
    old_adam = copy.deepcopy(restored.base.state_dict())
    restored.close()
    runtime = NativeOpal(restored.base, model, source=source, run_dir=run_dir,
                         recipe=recipe, backend=backend, agent_budget=agent_budget,
                         initial_program=program_dir)
    try:
        new_scheduler = copy.deepcopy(runtime.scheduler.state_dict())
        # Restore all custom updater fields too, not only its underlying Adam.
        programs = importlib.import_module('opal.programs')
        with programs.keep_param_group_identity():
            runtime.optimizer.updater.load_state_dict(expected['updater'])
        # Deliberate phase change: restore the NEW schedule after inheriting the
        # updater. Otherwise its load_state_dict silently puts the ended LR back.
        runtime.scheduler.load_state_dict(new_scheduler)
        for group, rate, initial in zip(runtime.base.param_groups,
                                       new_scheduler['_last_lr'], new_scheduler['base_lrs']):
            group['lr'], group['initial_lr'] = rate, initial
        actual_adam = runtime.base.state_dict()
        if not exact_state(old_adam['state'], actual_adam['state']):
            raise ValueError('Phase transition reset or altered Adam moments/counters')
        for old, new in zip(old_adam['param_groups'], actual_adam['param_groups']):
            if not exact_state({k:v for k,v in old.items() if k not in ('lr','initial_lr')},
                               {k:v for k,v in new.items() if k not in ('lr','initial_lr')}):
                raise ValueError('Undeclared Adam group change')
        if runtime.optimizer.files != expected['files'] or runtime.optimizer.update != 0:
            raise ValueError('New phase must inherit exact code and start at local update0')
        runtime.diagnostics_state = dict(parent['training_diagnostics'], last_review_step=0)
        restore_rng(before_rng)
        proof = dict(parent_local_updates=expected['update'], parent_program_version=expected['version'],
                     phase_local_update=0, phase_program_version=runtime.optimizer.version,
                     program_hashes={n:hashlib.sha256(t.encode()).hexdigest() for n,t in expected['files'].items()},
                     updater_fields_inherited=True, adam_moments_and_counters_exact=True,
                     rng_exact=exact_state(before_rng, rng_state()), new_optimizer_updates=0,
                     declared_changes=['new local OPAL/analysis clock; parent history archived',
                                       'new admitted warmup/cosine schedule',
                                       'diagnostic review index reset, parent TRAIN/DEV means retained'],
                     parent_checkpoint_preserved=True, training_authorized_by_this_helper=False)
        return runtime, proof
    except BaseException:
        runtime.close()
        raise
