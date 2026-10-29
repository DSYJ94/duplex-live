"""Restore complete wrapped native state without advancing an optimizer.

This helper grants no training or compute allowance. A native caller must verify
its authorization/lease and run independently before a new phase is admitted.
"""
import hashlib
from pathlib import Path

import torch

from .lora import adapter_state, adapter_parameters, load_adapter
from .native_optimizer import NativeOpal, assert_optimizer_equal
from .opal_adapter import UPSTREAM_COMMIT, checkpoint_manifest
from .state import exact_state, restore_rng, rng_state


def restore(path, expected_sha256, model, base_optimizer, *, source, run_dir):
    """Return the restored runtime and exact zero-update proof; caller closes it.

    Use a new output directory. Parent programs, scheduler, observations and Adam
    state are restored together; no scheduler reset/phase transition occurs here.
    """
    path, run_dir = Path(path), Path(run_dir)
    if run_dir.exists():
        raise ValueError('Fresh restore directory required; preserve previous attempts')
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_sha256:
        raise ValueError('Exact preserved checkpoint hash required')
    state = torch.load(path, map_location='cpu', weights_only=False)
    wrapped = state['optimizer']
    recipe = state['metadata']['recipe']
    if wrapped.get('opal') != 1 or wrapped.get('updater') is None:
        raise ValueError('Complete wrapped OPAL checkpoint required')
    if state['step'] != wrapped['update'] or not 0 < state['step'] <= recipe['updates']:
        raise ValueError('Checkpoint and OPAL progress differ')
    if recipe.get('opal', {}).get('upstream_commit') != UPSTREAM_COMMIT:
        raise ValueError('Exact pinned native OPAL recipe required')
    if state['rng']['cuda'] and not torch.cuda.is_available():
        raise ValueError('Native CUDA RNG cannot be verified by a CPU-only restore')
    if base_optimizer.state_dict()['state']:
        raise ValueError('Construct a fresh optimizer before complete state restoration')
    named = adapter_parameters(model)
    actual_parameters = [parameter for group in base_optimizer.param_groups for parameter in group['params']]
    if not named or [id(p) for p in actual_parameters] != [id(p) for p in named.values()]:
        raise ValueError('Original named adapter parameter order required for Adam state binding')
    initial_program = None
    if state['metadata'].get('initial_programs'):
        initial_program = run_dir.with_name(run_dir.name + '-initial-program')
        initial_program.mkdir(parents=True, exist_ok=False)
        for name, text in state['metadata']['initial_programs'].items():
            if Path(name).name != name or not name.endswith('.py'):
                raise ValueError('Only exact saved phase program files allowed')
            (initial_program/name).write_text(text)
    runtime = NativeOpal(base_optimizer, model, source=source, run_dir=run_dir,
                         recipe=recipe, backend='keep', agent_budget=0., initial_program=initial_program)
    try:
        load_adapter(model, state['adapter'])
        runtime.optimizer.load_state_dict(wrapped)
        runtime.restore_diagnostics(state['training_diagnostics'])
        restore_rng(state['rng'])
        actual = runtime.optimizer.state_dict()
        assert_optimizer_equal(wrapped, actual)
        if not exact_state(state['adapter'], adapter_state(model)):
            raise ValueError('Restored adapter differs')
        if not exact_state(state['rng'], rng_state()):
            raise ValueError('Restored RNG differs')
        if not exact_state(state['training_diagnostics'], runtime.diagnostics_state):
            raise ValueError('Restored diagnostic history differs')
        total = state['metadata'].get('parent_training_steps', 0) + state['step']
        counters = [int(value['step'].item()) for opt in actual['user_optimizers']
                    for value in opt['state'].values() if 'step' in value]
        if len(counters) != len(named) or any(value != total for value in counters):
            raise ValueError('Actual Adam counters differ from saved total training steps')
        proof = dict(checkpoint_sha256=actual_hash, adapter_exact=True,
                     numerical_program_history_state_exact=True, adam_exact=True, rng_exact=True,
                     diagnostics_exact=True, adam_counters=counters,
                     total_training_steps=total, **checkpoint_manifest(runtime.optimizer),
                     new_optimizer_updates=0, phase_schedule_changed=False,
                     external_calls=0, model_device=str(next(model.parameters()).device),
                     compared_state_exclusions=['run_dir', 'event_offset', 'backend'],
                     transport='keep-only; original backend state remains in preserved parent',
                     independent_process_established_by_caller=False,
                     native_behavior_qualified=False, training_authorized_by_this_helper=False)
        return runtime, proof
    except BaseException:
        runtime.close()
        raise
