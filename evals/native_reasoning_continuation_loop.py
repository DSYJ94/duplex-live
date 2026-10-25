"""Bounded continuation loop, without a CLI, model loader or provider mutation.

The separate runtime bridge must establish exact parent/execution/observer parity
before calling this library. No controller or authorization record is created
here. A successful loop is not a native restore proof or behavioral approval.
"""
import json
import math
from pathlib import Path
import time

from evals.native_reasoning_continuation_objective import RATE, START, UPDATES
from evals.native_reasoning_continuation_state import validate_metadata

SCHEMA = 'dl-native-changed-rate8-loop-v1'
from evals.native_reasoning_continuation_policy import (
    PROPOSAL_SHA, ORIGINAL_GRANT_SHA, RESOURCE_ID, authorization, check_guard)


def required_completion_seconds(completed_updates):
    """Conservative forecast allowances, not measured future runtimes.

    The old failed fit interval gives1022s per partition; the successful full
    independent restore took6126s. Preserve additional final32/native-speech/
    copyback allowances. See the committed phase-time evidence document.
    """
    if type(completed_updates) is not int or not 0 <= completed_updates <= UPDATES:
        raise ValueError('A bounded completed-update count is required')
    return (UPDATES - completed_updates) * 1022 + 2700 + 6126 + 3600 + 900


def _admit(checked, completed_updates):
    if checked['deadline_unix'] - time.time() < required_completion_seconds(completed_updates):
        raise ValueError('Insufficient remaining time for updates AND complete approval/copyback')
    return checked


def _write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _drive(operations, out, approved, guard):
    """Protocol seam for deterministic failure tests; no automatic retry."""
    first_guard = _admit(check_guard(guard, approved), 0)
    out = Path(out)
    out.mkdir()  # Existing/partial runs cannot be reused.
    rows = []
    returned = 0
    step_in_flight = False
    def event(phase, **fields):
        value = dict(phase=phase, at_unix=time.time(), **fields)
        with (out / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
        print(json.dumps(value, allow_nan=False), flush=True)
    event('started', authorization_sha256=approved['sha256'])
    try:
        for local_step in range(UPDATES):
            _admit(check_guard(guard, approved), local_step)
            row = operations.backward(local_step)
            _admit(check_guard(guard, approved), local_step)  # Backward may outlast a prior lease.
            gradient = operations.clip_and_check()
            row.update(gradient, learning_rate=RATE)
            _admit(check_guard(guard, approved), local_step)  # Last check immediately before mutation.
            event('optimizer_step_intent', local_step=local_step + 1,
                  expected_global_adam_step=START + local_step + 1)
            step_in_flight = True
            operations.step()
            step_in_flight = False
            returned += 1
            event('optimizer_step_returned', local_step=local_step + 1)
            row.update(operations.after_step(local_step))
            checkpoint = operations.checkpoint(out, local_step + 1, row)
            rows.append(row)
            event('checkpoint_closed', local_step=local_step + 1,
                  global_adam_step=START + local_step + 1, checkpoint=checkpoint)
        exposure = operations.finish(rows)
        result = dict(schema=SCHEMA, phase='eight_updates_closed_not_qualified',
            authorization_sha256=approved['sha256'], initial_guard=first_guard,
            optimizer_step_calls_returned=returned, closed_checkpoints=len(rows),
            rows=rows, exposure=exposure, native_restore_approved=False,
            behavior_qualified=False, model_adopted=False)
        _write(out / 'training-record.json', result)
        return result
    except BaseException as error:
        # Never infer that a failed optimizer call did not partially mutate state.
        # Preserve intent, returned-call and closed-checkpoint counts separately.
        event('failed_no_retry', error_type=type(error).__name__,
              optimizer_step_calls_returned=returned, closed_checkpoints=len(rows),
              optimizer_state_requires_audit=step_in_flight or returned != len(rows))
        raise


class NativeOperations:
    """Real native backward/clip/Adam/checkpoint operations, never a test backend."""
    def __init__(self, context, optimizer, metadata):
        import torch
        from evals.moshi_training_smoke import tensor_hashes
        from evals.native_reasoning_continuation_state import _adam
        from evals.context_execution import snapshot
        validate_metadata(metadata)
        self.context, self.optimizer, self.metadata = context, optimizer, metadata
        self.learned = context['learned']
        self.execution = snapshot(context['lm'])
        if any(self.execution[k] != v for k, v in context['control_policy']['execution_required'].items()):
            raise ValueError('Original independently approved native execution profile required')
        _adam(optimizer, START, RATE)
        if ({n for n, p in context['lm'].named_parameters() if p.requires_grad} != set(self.learned)
                or any(p.grad is not None for p in context['lm'].parameters())):
            raise ValueError('Only the exact26 clean native adapters may be trainable')
        self.frozen = tensor_hashes((n, p) for n, p in context['lm'].named_parameters() if not p.requires_grad)
        if self.frozen != context['frozen'] or not torch.is_grad_enabled():
            raise ValueError('Original frozen base and native autograd execution required')

    def backward(self, local_step):
        from evals.native_reasoning_continuation_objective import backward
        self.optimizer.zero_grad(set_to_none=True)
        return backward(self.context, local_step)

    def clip_and_check(self):
        import torch
        if (any(p.grad is None or not torch.isfinite(p.grad).all() for p in self.learned.values())
                or any(p.grad is not None for n, p in self.context['lm'].named_parameters() if n not in self.learned)):
            raise ValueError('All26 finite gradients and zero frozen gradients required')
        counts = dict(
            nonzero_A_gradients=sum(int(torch.count_nonzero(p.grad)) for n, p in self.learned.items() if '.lora_A.' in n),
            nonzero_B_gradients=sum(int(torch.count_nonzero(p.grad)) for n, p in self.learned.items() if '.lora_B.' in n))
        norm = float(torch.nn.utils.clip_grad_norm_(list(self.learned.values()), 1.))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('Finite nonzero gradient norm required')
        return dict(all26_gradients_finite=True, frozen_parameter_gradients=0,
                    **counts,
                    pre_clip_gradient_norm=norm, gradient_clip_norm=1.)

    def step(self):
        self.before = {n:p.detach().clone() for n, p in self.learned.items()}
        self.optimizer.step()

    def after_step(self, local_step):
        import torch
        from evals.native_reasoning_continuation_state import _adam
        _adam(self.optimizer, START + local_step + 1, RATE)
        delta = sum(float((p.detach().double() - self.before[n].double()).square().sum())
                    for n, p in self.learned.items()) ** .5
        del self.before
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError('Finite nonzero actual adapter displacement required')
        self.optimizer.zero_grad(set_to_none=True)
        return dict(parameter_delta_l2=delta)

    def checkpoint(self, out, local_steps, row):
        from evals.native_reasoning_continuation_state import save_update
        return save_update(out, self.learned, self.optimizer, local_steps,
                           self.metadata, step_record=row)

    def finish(self, rows):
        import torch
        from evals.moshi_training_smoke import tensor_hashes
        from evals.context_execution import snapshot
        from evals.native_reasoning_continuation_objective import validate_stage
        if self.frozen != tensor_hashes((n, p) for n, p in self.context['lm'].named_parameters() if not p.requires_grad):
            raise ValueError('Frozen base changed during continuation')
        peak = torch.cuda.max_memory_allocated()
        if snapshot(self.context['lm']) != self.execution or not 0 < peak < 80_000_000_000:
            raise ValueError('Changed execution profile or exceeded original80GB bound')
        result = validate_stage(rows, self.context['audio49_policy'],
                                self.context['control_policy'], self.context['reasoning_policy'])
        return dict(result, peak_allocated_bytes=peak, native_execution_profile=self.execution)


def run(context, optimizer, metadata, out, authorization_path, guard):
    """Caller must first establish parent numerical parity in the pinned runtime."""
    approved = authorization(authorization_path)
    validate_metadata(metadata)
    if metadata['continuation']['authorization_sha256'] != approved['sha256']:
        raise ValueError('State provenance and physical authorization differ')
    check_guard(guard, approved)
    return _drive(NativeOperations(context, optimizer, metadata), out, approved, guard)
