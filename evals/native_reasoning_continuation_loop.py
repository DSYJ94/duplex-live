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
from evals.native_reasoning_continuation_state import _sha, validate_metadata

SCHEMA = 'dl-native-changed-rate8-loop-v1'
PROPOSAL_SHA = '750f5886f9f6d90fb17eabadb80f99e41b9e1c420da9a8929faaab942113c40a'
ORIGINAL_GRANT_SHA = '97c8d43fe7be11595e7d00775af4e1f2c7666a469041c720bf7586ee8a020322'
RESOURCE_ID = '4443302983961393208'


def authorization(path):
    """Read a NEW direct-owner authorization; the old zero-update grant fails."""
    path = Path(path)
    record = json.loads(path.read_text())
    expected = dict(schema='dl-native-changed-rate-eight-owner-authorization-v1',
        status='APPROVED', approval_source='Direct human reply in active LOCAL chat',
        proposal_sha256=PROPOSAL_SHA, original_grant_sha256=ORIGINAL_GRANT_SHA,
        project='aurium-duplex-live-20261024', additional_optimizer_updates=8,
        initial_global_adam_step=528, final_global_adam_step=536,
        learning_rate=1e-6, maximum_additional_gcp_usage_usd=50,
        cumulative_gcp_usage_cap_usd=150, additional_gpu_hours=6,
        cumulative_gpu_hours=18, maximum_concurrent_gpus=1,
        rejected_parent_bounded_exploration_approved=True,
        new_credit_purchase=False, new_vm_or_storage=False,
        further_extension_authorized=False, model_adoption_authorized=False)
    if (any(record.get(k) != v or type(record.get(k)) is not type(v)
            for k, v in expected.items())
            or type(record.get('owner_reply')) is not str or not record['owner_reply'].strip()):
        raise ValueError('Exact separate changed-rate/update/budget owner authorization required')
    return dict(record=record, sha256=_sha(path))


def check_guard(guard, approved):
    raw = guard()
    from portable.context.native_saved_stage_runtime import guard_receipt
    checked = guard_receipt(lambda: raw)
    now = time.time()
    numeric = ('native_start_unix', 'maximum_new_gpu_seconds', 'hourly_price_usd',
               'prior_gpu_seconds_upper', 'prior_scope_cost_upper_usd')
    if (any(type(raw.get(k)) not in (int, float) or not math.isfinite(raw[k]) for k in numeric)
            or checked['provider'] != 'gcp' or checked['resource_id'] != RESOURCE_ID
            or checked['authorization_sha256'] != approved['sha256']
            or raw.get('optimizer_updates_authorized') != 8
            or raw.get('parent_global_adam_step') != 528
            or raw.get('final_global_adam_step_limit') != 536
            or not 0 < raw['native_start_unix'] <= now
            or not 0 < raw['maximum_new_gpu_seconds'] <= 21600
            or checked['deadline_unix'] > raw['native_start_unix'] + raw['maximum_new_gpu_seconds']
            or raw['prior_gpu_seconds_upper'] < 40351
            or raw['prior_gpu_seconds_upper'] + raw['maximum_new_gpu_seconds'] > 64800
            or raw['prior_scope_cost_upper_usd'] < 91.98448055555555
            or checked['planned_maximum_cost_usd'] > 50
            or checked['remaining_budget_usd'] > 50
            or not 0 < raw['hourly_price_usd'] <= 7.25
            or checked['planned_maximum_cost_usd'] < raw['hourly_price_usd'] * raw['maximum_new_gpu_seconds'] / 3600 + 6.5
            or checked['planned_maximum_cost_usd'] + raw['prior_scope_cost_upper_usd'] > 150):
        raise ValueError('Fresh unchanged provider/budget/update bounds required')
    return checked


def _write(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _drive(operations, out, approved, guard):
    """Protocol seam for deterministic failure tests; no automatic retry."""
    first_guard = check_guard(guard, approved)
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
            check_guard(guard, approved)
            row = operations.backward(local_step)
            check_guard(guard, approved)  # Backward may outlast a prior lease.
            gradient = operations.clip_and_check()
            row.update(gradient, learning_rate=RATE)
            check_guard(guard, approved)  # Last check immediately before mutation.
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
        validate_metadata(metadata)
        self.context, self.optimizer, self.metadata = context, optimizer, metadata
        self.learned = context['learned']
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
        from evals.moshi_training_smoke import tensor_hashes
        from evals.native_reasoning_continuation_objective import validate_stage
        if self.frozen != tensor_hashes((n, p) for n, p in self.context['lm'].named_parameters() if not p.requires_grad):
            raise ValueError('Frozen base changed during continuation')
        return validate_stage(rows, self.context['audio49_policy'],
                              self.context['control_policy'], self.context['reasoning_policy'])


def run(context, optimizer, metadata, out, authorization_path, guard):
    """Caller must first establish parent numerical parity in the pinned runtime."""
    approved = authorization(authorization_path)
    validate_metadata(metadata)
    if metadata['continuation']['authorization_sha256'] != approved['sha256']:
        raise ValueError('State provenance and physical authorization differ')
    check_guard(guard, approved)
    return _drive(NativeOperations(context, optimizer, metadata), out, approved, guard)
