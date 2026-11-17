"""Restore learned action state without updates; native efficacy is separate.

The caller establishes different-process provenance and pinned checkpoint hashes.
This seam checks parameter ownership and every saved Adam/OPAL/RNG field, and
works on the actual RetReadout.delta as well as a standalone cached-fit head.
"""
from .action_inference import validate_fit


def restore_fit(state, delta, runtime, expected_identity):
    from training.venus.native_optimizer import assert_optimizer_equal
    from training.venus.opal_adapter import checkpoint_manifest
    from training.venus.state import exact_state, restore_rng, rng_state

    meta = validate_fit(state, expected_identity, delta.in_features)
    expected = [delta.weight, delta.bias] if meta['mode'] == 'readout' else [delta.bias]
    actual = [p for group in runtime.base.param_groups for p in group['params']]
    if len(actual) != len(expected) or any(a is not b for a, b in zip(actual, expected)):
        raise ValueError('Optimizer must own exactly the selected head parameters')
    if any(not p.requires_grad for p in expected):
        raise ValueError('Restore the trainable head before freezing native inference')
    if meta['mode'] == 'bias' and delta.weight.requires_grad:
        raise ValueError('Scalar control must keep hidden weights frozen')
    delta.load_state_dict(state['delta'], strict=True)
    runtime.optimizer.load_state_dict(state['optimizer'])
    runtime.restore_diagnostics(state['diagnostics'])
    restore_rng(state['rng'])
    assert_optimizer_equal(state['optimizer'], runtime.optimizer.state_dict())
    if not exact_state(delta.state_dict(), state['delta']):
        raise ValueError('Learned action coefficients differ after restore')
    if not exact_state(runtime.diagnostics_state, state['diagnostics']):
        raise ValueError('OPAL diagnostics differ after restore')
    if not exact_state(rng_state(), state['rng']):
        raise ValueError('Action RNG differs after restore')
    if not exact_state(checkpoint_manifest(runtime.optimizer), state['program']):
        raise ValueError('Complete saved OPAL updater manifest differs')
    return dict(saved_readout_updates=state['step'], new_optimizer_updates=0,
                Adam_RNG_OPAL_exact=True, native_free_running_qualified=False)
