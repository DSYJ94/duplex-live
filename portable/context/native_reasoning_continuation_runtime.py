"""Fresh-process bridge for the proposed changed saved528 continuation.

Library only. No VM start, container launch, CLI, budget approval or auto retry.
An external controller must own native STOP, guards, terminal records and copyback.
"""
import fcntl
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from evals.native_reasoning_continuation_policy import authorization, check_guard
from evals.native_reasoning_saved_stage_audit import SOURCE
from evals.native_saved_stage_context import bind
from evals.native_saved_stage_recovery import verified_inputs, process_identity, canonical
from portable.context.native_saved_stage_runtime import (
    PURE_HELPERS, arguments, git, restore_recorded_precision)

SCHEMA = 'dl-native-changed-rate8-runtime-record-v1'


def _isolate(source, producer):
    """Keep original model/training modules first; add only absent new helpers."""
    source = Path(source).resolve(strict=True)
    if git(source, 'rev-parse', 'HEAD') != SOURCE or git(source, 'status', '--porcelain'):
        raise ValueError('Clean original DA model checkout required')
    allowed = PURE_HELPERS | {'evals.native_reasoning_continuation_policy'}
    unexpected = [name for name in sys.modules
        if (name.startswith('evals.') and name not in allowed)
        or name == 'torch' or name == 'contextspan'
        or name.startswith(('contextspan.', 'training.'))]
    if unexpected:
        raise ValueError('Fresh process required before continuation model construction')
    import evals
    evals.__path__ = [str(source / 'evals'), str(producer / 'evals')]
    for name, relative in (('portable', 'portable'), ('portable.context', 'portable/context')):
        if name in sys.modules:
            sys.modules[name].__path__ = [str(source / relative), str(producer / relative)]
    sys.path[:0] = [str(source), str(producer)]
    os.chdir(source)
    os.environ['DL_TASK_COMMIT'] = SOURCE
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    return source


def _write(path, record):
    with path.open('x') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')


def _independent_identity(first, second):
    from evals.native_saved_stage_proof import identity
    if (not identity(first) or not identity(second) or first['pid'] == second['pid']
            or first['birth'] == second['birth']
            or any(first[k] != second[k] for k in ('host', 'boot_id'))):
        raise ValueError('Distinct fresh processes on the same recorded runtime required')


def execute(source, snapshot, observation, out, producer_commit, authorization_path,
            guard, parent_proof, *, phase='fit', reference=None):
    """Exactly one fit OR zero-update independent restore per new process."""
    authorization_path = Path(authorization_path).resolve(strict=True)
    approved = authorization(authorization_path)
    initial_guard = check_guard(guard, approved)
    if phase not in ('fit', 'independent') or (reference is None) != (phase == 'fit'):
        raise ValueError('One new fit or a separately referenced independent restore required')
    producer = Path(__file__).resolve().parents[2]
    if git(producer, 'rev-parse', 'HEAD') != producer_commit or git(producer, 'status', '--porcelain'):
        raise ValueError('Exact clean committed continuation producer required')
    out = Path(out).resolve()
    snapshot = Path(snapshot).resolve(strict=True)
    observation = Path(observation).resolve(strict=True)
    parent_proof = Path(parent_proof).resolve(strict=True)
    if reference is not None:
        reference = dict(reference, path=str(Path(reference['path']).resolve(strict=True)))
    source = _isolate(source, producer)
    inputs = verified_inputs(snapshot, observation)
    roots = [Path('/projects/u6si/duplex-live/runs').resolve(), source / 'runs', inputs['root']]
    if out.exists() or any(out == root or root in out.parents or out in root.parents for root in roots):
        raise ValueError('Separate unused output outside every immutable input root required')
    lock = (out.parent / 'native-control40-gpu-owner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise ValueError('Another GPU process is active')
    from evals.native_reasoning_continuation_observation import (
        load_observer, parent_reference, observe, compare)
    from evals.native_reasoning_continuation_state import (
        restore_parent, restore_final, stage_metadata, _sha)
    from evals.native_reasoning_continuation_loop import run
    expected_parent = parent_reference(parent_proof)
    observer = load_observer(producer)
    previous = None
    if reference is not None:
        path = Path(reference['path']).resolve(strict=True)
        if _sha(path) != reference['sha256']:
            raise ValueError('Exact closed fit record bytes required')
        previous = json.loads(path.read_text())
        if (previous['schema'] != SCHEMA or previous['phase'] != 'fit'
                or previous['producer_commit'] != producer_commit
                or previous['authorization_sha256'] != approved['sha256']
                or previous['optimizer_updates_this_process'] != 8):
            raise ValueError('Completed distinct changed-stage fit reference required')
    args = arguments(inputs['controller'], out)
    from portable.context.native_training_cached import verify_receipt, CODE, sha, PROFILE
    ready = Path(args[args.index('--native-audio49-preflight') + 1])
    verify_receipt(ready / 'immutable-input-readiness.json', 'readiness', SOURCE)
    module = importlib.import_module('evals.context_actual_train')
    if Path(module.__file__).resolve() != source / 'evals/context_actual_train.py':
        raise ValueError('Original model constructor resolved outside pinned source')
    captured = []
    prefix = None
    def live_guard():
        return check_guard(guard, approved)
    def capture(context):
        import torch
        from evals.native_saved_stage_recovery import validate_context
        if captured or prefix['derived_ast_sha256'] != '7907f6de7ddbdaa1a22111829a67cfb1a4a2e5df629482f1693c4676549d15f0':
            raise ValueError('Only the exact optimizer-free original construction prefix is allowed')
        live_guard()
        parent = validate_context(context, inputs)
        metadata = stage_metadata(parent, producer_commit, approved['sha256'])
        if phase == 'fit':
            optimizer, restored_metadata, receipt = restore_parent(
                context, inputs, producer_commit, approved['sha256'])
            if restored_metadata != metadata:
                raise ValueError('Restored continuation recipe differs')
            _write(out / 'parent-state-receipt.json', receipt)
            initial = observe(context, optimizer, observer, live_guard, parent=expected_parent)
            _write(out / 'parent-observation.json', initial)
            result = run(context, optimizer, metadata, out / 'continuation-fit',
                         authorization_path, guard)
            final = observe(context, optimizer, observer, live_guard)
            checkpoint = out / 'continuation-fit/update-008'
            evidence = dict(loop_record_sha256=_sha(out / 'continuation-fit/training-record.json'),
                closed_checkpoints=result['closed_checkpoints'],
                initial_observation_sha256=_sha(out / 'parent-observation.json'))
        else:
            _independent_identity(previous['process_identity'], process_identity())
            if previous['metadata'] != metadata or previous['context_prefix'] != prefix:
                raise ValueError('Independent source/recipe/construction differs from fit')
            checkpoint = Path(previous['final_checkpoint']).resolve(strict=True)
            if _sha(checkpoint / 'checkpoint.json') != previous['checkpoint_receipt_sha256']:
                raise ValueError('Closed fit checkpoint identity changed')
            optimizer = torch.optim.AdamW(list(context['learned'].values()), lr=1e-6, weight_decay=0.)
            state = restore_final(checkpoint, context['learned'], optimizer, metadata)
            final = observe(context, optimizer, observer, live_guard)
            compare(final, previous['final_observation'])
            evidence = dict(fit_reference_sha256=reference['sha256'], state_restore=state,
                fresh_process_all_fingerprints_and_losses_exact=True)
        _write(out / 'final-observation.json', final)
        record = dict(schema=SCHEMA, phase=phase, model_source_commit=SOURCE,
            producer_commit=producer_commit, authorization_sha256=approved['sha256'],
            context_prefix=prefix, process_identity=process_identity(), metadata=metadata,
            optimizer_updates_this_process=8 if phase == 'fit' else 0,
            final_global_adam_step=536, final_checkpoint=str(checkpoint),
            checkpoint_receipt_sha256=_sha(checkpoint / 'checkpoint.json'),
            final_observation=final, evidence=evidence,
            actual_terminal_verified=False, numerical_proof_approved=False,
            behavior_qualified=False, model_adopted=False)
        live_guard()
        _write(out / 'continuation-record.json', record)
        captured.append(record)
        return record
    function, prefix = bind(Path(module.__file__).read_bytes(), vars(module), capture)
    module.run = function
    from evals.immutable_file_hashes import verified_reads
    sys.argv = ['changed-continuation-' + phase, *args]
    live_guard()
    precision = restore_recorded_precision(importlib.import_module('torch'))
    passed = False
    reads = None
    try:
        with verified_reads(roots[:2]) as reads:
            importlib.import_module('evals.context_training_proof').main()
        if len(captured) != 1:
            raise ValueError('Original construction did not reach the new continuation phase')
        if git(producer, 'status', '--porcelain') or git(source, 'status', '--porcelain'):
            raise ValueError('Original or producer source changed during execution')
        live_guard()
        passed = True
    finally:
        if reads is not None and out.is_dir():
            _write(out / 'continuation-runtime-inputs.json', dict(schema=SCHEMA,
                phase=phase, runtime_completed=passed, model_source_commit=SOURCE,
                producer_commit=producer_commit, input_reads=reads.receipt(),
                profile=PROFILE, input_code_sha256={name:sha(source / name) for name in CODE},
                precision=precision, context_prefix=prefix, initial_guard=initial_guard,
                record_sha256=canonical(captured[0]) if passed else None,
                actual_terminal_verified=False, numerical_proof_approved=False,
                behavior_qualified=False, model_adopted=False))
    return captured[0]
