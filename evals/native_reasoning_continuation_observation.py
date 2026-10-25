"""Read every native fingerprint and retain losses without another model pass."""
import importlib.util
import inspect
import json
from pathlib import Path
import random
from copy import deepcopy


def load_observer(producer):
    """Load the new observer separately; never replace the original trainer module."""
    path = Path(producer) / 'evals/native_reasoning_training.py'
    spec = importlib.util.spec_from_file_location('_duplex_continuation_observer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = module.fingerprints
    parameter = inspect.signature(function).parameters.get('metrics')
    if parameter is None or parameter.kind != inspect.Parameter.KEYWORD_ONLY:
        raise ValueError('Separately prepared same-forward endpoint observer required')
    return function


def parent_reference(proof):
    from portable.context.native_matched_reasoning_loss import pinned_proof, sha
    report = pinned_proof(proof)
    path = Path(proof) / 'independent-record.json'
    if sha(path) != report['recovery_record_sha256'][path.name]:
        raise ValueError('Exact completed independent parent record required')
    record = json.loads(path.read_text())
    if (record['full_independent_restore_verified'] is not True
            or record['optimizer_and_rng_exact'] is not True
            or record['optimizer_updates_during_recovery'] != 0):
        raise ValueError('Full independently restored saved528 parent required')
    return record['evidence']


def compare(actual, expected):
    """Compare measured fields only; process/phase identities must remain distinct."""
    keys = ('all_fingerprints', 'original28_metrics', 'reasoning32_metrics', 'reasoning32_summary')
    if any(actual[key] != expected[key] for key in keys):
        raise ValueError('Complete native fingerprints or endpoint losses differ on restoration')
    return True


def observe(context, optimizer, observer, guard, *, parent=None):
    """No updates or backwards. Always return RNG to the pre-observation state."""
    import numpy as np
    import torch
    from training.control.train_context import assert_equal, cpu_tree
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_native_audio49_training import fingerprints as acoustic
    from evals.context_native_stream_training import fingerprints as streaming
    from evals.native_control_training import fingerprints as control
    from evals.native_saved_stage_recovery import validate_complete_fingerprints, process_identity
    from evals.native_saved_stage_proof import validate_metrics
    from portable.context.native_matched_reasoning_loss import summarize
    lm = context['lm']
    learned = context['learned']
    if any(p.grad is not None for p in lm.parameters()):
        raise ValueError('Observation requires zero accumulated gradients')
    installed = tensor_hashes(learned.items())
    frozen = tensor_hashes((n, p) for n, p in lm.named_parameters() if not p.requires_grad)
    optimizer_before = deepcopy(cpu_tree(optimizer.state_dict()))
    numpy = np.random.get_state()
    numpy = (numpy[0], numpy[1].copy(), *numpy[2:])
    rng = (random.getstate(), numpy, torch.get_rng_state(), torch.cuda.get_rng_state_all())
    metrics = []
    fingerprints = {}
    try:
        with torch.no_grad():
            guard()
            original_metrics = context['metrics']()
            fingerprints['original28'] = context['digests']()
            guard()
            fingerprints['audio49'] = acoustic(lm, context['audio49_batches'])
            guard()
            fingerprints['native49'] = streaming(lm, context['stream_bound'])
            guard()
            fingerprints['control40'] = control(lm, context['control_bound'])
            guard()
            fingerprints['reasoning32'] = observer(lm, context['reasoning_bound'], metrics=metrics)
        metadata = context['state_metadata'](context['frozen'])
        validate_complete_fingerprints(fingerprints, metadata,
                                      dict(native49={}, control40={}, reasoning32={}))
        validate_metrics(original_metrics, [r['scenario'] for r in context['audits']])
        summary = summarize(metrics, fingerprints['reasoning32'])
        if parent is not None:
            assert_equal(parent['all_fingerprints'], fingerprints)
            assert_equal(parent['final_metrics_recomputed'], original_metrics)
        if (installed != tensor_hashes(learned.items())
                or frozen != context['frozen']
                or frozen != tensor_hashes((n, p) for n, p in lm.named_parameters() if not p.requires_grad)
                or any(p.grad is not None for p in lm.parameters())):
            raise ValueError('Observation changed adapters, frozen base or gradient ownership')
        assert_equal(optimizer_before, cpu_tree(optimizer.state_dict()))
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
    assert_equal(rng[0], random.getstate())
    assert_equal(rng[2], torch.get_rng_state())
    assert_equal(rng[3], torch.cuda.get_rng_state_all())
    n = np.random.get_state()
    assert_equal((rng[1][0], rng[1][1].tolist(), *rng[1][2:]),
                 (n[0], n[1].tolist(), *n[2:]))
    peak = torch.cuda.max_memory_allocated()
    if not 0 < peak < 80_000_000_000:
        raise ValueError('Native observation must fit the original80GB memory bound')
    guard()
    return dict(schema='dl-native-continuation-observation-v1',
        process_identity=process_identity(), all_fingerprints=fingerprints,
        original28_metrics=original_metrics, reasoning32_metrics=metrics,
        reasoning32_summary=summary, all_state_unchanged=True, rng_restored=True,
        peak_allocated_bytes=peak,
        optimizer_updates_during_observation=0, backward_calls=0,
        native_restore_approved=False, behavior_qualified=False, model_adopted=False)
