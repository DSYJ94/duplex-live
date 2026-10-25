"""State primitives for a separately authorized saved528 continuation.

No CLI, model construction, optimizer step or resource creation. These helpers
do not authorize execution. A new runtime must verify the owner authorization,
provider/cost guards, source, parent proof and execution parity independently.
Old metadata/restoration rules are never relaxed to accept the new stage.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from evals.native_reasoning_continuation_objective import RATE, START, UPDATES
from evals.native_reasoning_saved_stage_audit import PILOT, PINNED, SOURCE
from evals.native_saved_stage_recovery import METADATA, canonical, validate_context

STATE_SHA = PINNED[PILOT + '/training/training-state.pt']
ADAPTER_SHA = PINNED[PILOT + '/training/adapters.safetensors']
PROOF_SHA = '27a6676dd95ef8100e5211b6e24ddca7e9b716c0cbcae4d3e8fb02215d9ccac7'
SCHEMA = 'dl-native-saved528-continuation-state-v1'


def _hex(value, length):
    if (type(value) is not str or len(value) != length
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Exact source or authorization digest required')
    return value


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stage_metadata(parent, producer_commit, authorization_sha256):
    """Name the changed recipe explicitly; a digest alone is not permission."""
    if canonical(parent) != METADATA or parent['repo_commit'] != SOURCE:
        raise ValueError('Exact original saved528 metadata required')
    _hex(producer_commit, 40)
    _hex(authorization_sha256, 64)
    if producer_commit == SOURCE or parent['recipe']['lr'] != 2e-7:
        raise ValueError('A separate committed continuation producer is required')
    result = deepcopy(parent)
    result['repo_commit'] = producer_commit
    result['recipe']['lr'] = RATE
    result['continuation'] = dict(schema=SCHEMA, parent_repo_commit=SOURCE,
        parent_metadata_sha256=METADATA, parent_state_sha256=STATE_SHA,
        parent_adapter_sha256=ADAPTER_SHA, parent_proof_sha256=PROOF_SHA,
        authorization_sha256=authorization_sha256, initial_global_adam_step=START,
        additional_updates=UPDATES, final_global_adam_step=START + UPDATES,
        objective_implementation_sha256=_sha(Path(__file__).with_name('native_reasoning_continuation_objective.py')),
        state_implementation_sha256=_sha(Path(__file__)),
        completed_steps_scope='Local updates in this NEW stage; Adam clocks remain global',
        change='Only optimizer group learning rate2e-7 ->1e-6; all moments/RNG retained',
        original_fit_actual_exit=-15, model_adopted=False)
    return result


def validate_metadata(metadata):
    """Reverse the declared edits and require the exact pinned parent."""
    parent = deepcopy(metadata)
    provenance = parent.pop('continuation', {})
    parent['repo_commit'] = SOURCE
    parent['recipe']['lr'] = 2e-7
    expected = stage_metadata(parent, metadata['repo_commit'],
                              provenance.get('authorization_sha256'))
    if metadata != expected:
        raise ValueError('Only the declared source/provenance/rate transition is allowed')
    return True


def _adam(optimizer, expected_clock, expected_rate):
    if (len(optimizer.param_groups) != 1 or len(optimizer.state) != 26
            or len(optimizer.param_groups[0]['params']) != 26):
        raise ValueError('All26 named parameters in one original AdamW group required')
    group = optimizer.param_groups[0]
    if (group['lr'] != expected_rate or group['betas'] != (.9, .999)
            or group['eps'] != 1e-8 or group['weight_decay'] != 0.):
        raise ValueError('Changed continuation Adam recipe')
    for state in optimizer.state.values():
        if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                or float(state['step']) != expected_clock):
            raise ValueError('Every Adam clock must match the exact global update')


def _exact(snapshot, learned, optimizer, local_steps, global_clock, rate_override=None):
    import numpy as np
    import torch
    from training.control.train_context import assert_equal, cpu_tree
    if (type(snapshot['completed_steps']) is not int
            or snapshot['completed_steps'] != local_steps
            or snapshot['parameter_names'] != list(learned)
            or len(learned) != 26 or sum(p.numel() for p in learned.values()) != 1894400
            or snapshot['scheduler'] is not None
            or any(p.grad is not None for p in learned.values())):
        raise ValueError('Exact named adapter coverage, stage counter and clean gradients required')
    expected = deepcopy(snapshot['optimizer'])
    rate = 2e-7 if rate_override is None else rate_override
    if rate_override is not None:
        if len(expected['param_groups']) != 1:
            raise ValueError('Only one original optimizer group can change rate')
        expected['param_groups'][0]['lr'] = rate_override
    assert_equal(snapshot['adapters'], cpu_tree(learned))
    assert_equal(expected, cpu_tree(optimizer.state_dict()))
    _adam(optimizer, global_clock, rate)
    if any(not torch.isfinite(v).all() for row in optimizer.state.values() for v in row.values()):
        raise ValueError('Nonfinite continuation optimizer state')
    if any(bool((row['exp_avg_sq'] < 0).any()) for row in optimizer.state.values()):
        raise ValueError('Invalid negative Adam second moment')
    assert_equal(snapshot['torch_cpu_rng'], torch.get_rng_state())
    assert_equal(snapshot['torch_cuda_rng'], torch.cuda.get_rng_state_all())
    assert_equal(snapshot['python_rng'], random.getstate())
    n = np.random.get_state()
    assert_equal(snapshot['numpy_rng'], dict(name=n[0], keys=n[1].tolist(),
        position=int(n[2]), has_gauss=int(n[3]), cached_gauss=float(n[4])))


def restore_parent(context, inputs, producer_commit, authorization_sha256):
    """Restore and compare EVERYTHING before changing only the group rate."""
    parent = validate_context(context, inputs)
    metadata = stage_metadata(parent, producer_commit, authorization_sha256)
    root = inputs['training']
    if (_sha(root / 'training-state.pt') != STATE_SHA
            or _sha(root / 'adapters.safetensors') != ADAPTER_SHA):
        raise ValueError('Pinned saved528 state or adapter bytes changed')
    # No Torch import or optimizer construction until physical inputs match.
    import torch
    from safetensors.torch import load_file
    from evals.context_training_state import restore
    from training.control.train_context import assert_equal
    learned = context['learned']
    if any(p.grad is not None for p in context['lm'].parameters()):
        raise ValueError('Parent restoration cannot inherit stale gradients')
    optimizer = torch.optim.AdamW(list(learned.values()), lr=2e-7, weight_decay=0.)
    snapshot = restore(root / 'training-state.pt', learned, optimizer, parent)
    assert_equal(snapshot['adapters'], load_file(root / 'adapters.safetensors', device='cpu'))
    _exact(snapshot, learned, optimizer, 8, START)
    optimizer.param_groups[0]['lr'] = RATE
    _exact(snapshot, learned, optimizer, 8, START, rate_override=RATE)
    return optimizer, metadata, dict(schema=SCHEMA, parent_state_sha256=STATE_SHA,
        parent_adapter_sha256=ADAPTER_SHA, global_adam_step=START,
        adapter_moments_counters_rng_exact_before_rate_change=True,
        only_group_lr_changed=True, optimizer_updates=0,
        numerical_restore_approved=False, model_adopted=False)


def save_update(out, learned, optimizer, local_steps, metadata):
    """Keep every update separately; publish the receipt after both files exist.

    An interrupted/incomplete directory is preserved and cannot be overwritten.
    Its existence never counts as a successful checkpoint or terminal record.
    """
    if type(local_steps) is not int or not 1 <= local_steps <= UPDATES:
        raise ValueError('Exactly local update1..8 required')
    validate_metadata(metadata)
    _adam(optimizer, START + local_steps, RATE)
    if any(p.grad is not None for p in learned.values()):
        raise ValueError('Clear accumulated gradients before committing a checkpoint')
    folder = Path(out) / ('update-' + str(local_steps).zfill(3))
    if folder.exists():
        raise FileExistsError('A closed or interrupted update cannot be overwritten')
    from evals.context_training_state import save
    from safetensors.torch import save_file
    folder.mkdir()  # Neither create missing parents nor reuse a partial update.
    state = save(folder / 'training-state.pt', learned, optimizer, local_steps, metadata)
    save_file({n: p.detach().cpu().contiguous() for n, p in learned.items()},
              folder / 'adapters.safetensors.partial')
    (folder / 'adapters.safetensors.partial').replace(folder / 'adapters.safetensors')
    _exact(state, learned, optimizer, local_steps, START + local_steps, rate_override=RATE)
    receipt = dict(schema=SCHEMA, local_completed_steps=local_steps,
        global_adam_step=START + local_steps, metadata_sha256=canonical(metadata),
        files={name: _sha(folder / name) for name in ('training-state.pt', 'adapters.safetensors')},
        independent_restore_approved=False, behavior_qualified=False, model_adopted=False)
    with (folder / 'checkpoint.json.partial').open('x') as stream:
        stream.write(json.dumps(receipt, indent=2) + '\n')
    (folder / 'checkpoint.json.partial').replace(folder / 'checkpoint.json')
    return receipt


def restore_final(folder, learned, optimizer, metadata):
    """Compare the closed final state; logits/fresh-process proof remain separate."""
    validate_metadata(metadata)
    folder = Path(folder)
    receipt = json.loads((folder / 'checkpoint.json').read_text())
    if (receipt['schema'] != SCHEMA or type(receipt['local_completed_steps']) is not int
            or receipt['local_completed_steps'] != UPDATES
            or receipt['global_adam_step'] != START + UPDATES
            or receipt['metadata_sha256'] != canonical(metadata)
            or set(receipt['files']) != {'training-state.pt', 'adapters.safetensors'}
            or any(_sha(folder / name) != value for name, value in receipt['files'].items())):
        raise ValueError('Complete closed local8/global536 checkpoint required')
    from evals.context_training_state import restore
    from safetensors.torch import load_file
    from training.control.train_context import assert_equal
    snapshot = restore(folder / 'training-state.pt', learned, optimizer, metadata)
    assert_equal(snapshot['adapters'], load_file(folder / 'adapters.safetensors', device='cpu'))
    _exact(snapshot, learned, optimizer, UPDATES, START + UPDATES, rate_override=RATE)
    return dict(schema=SCHEMA, local_completed_steps=UPDATES,
        global_adam_step=START + UPDATES, adapter_adam_rng_exact=True,
        native_logits_approved=False, independent_process_verified=False,
        behavior_qualified=False, model_adopted=False)
