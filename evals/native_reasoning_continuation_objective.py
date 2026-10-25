"""Objective composition for a proposed saved528 -> 536 experiment.

Library only: no optimizer, resource allocation, checkpoint write or CLI. The
authorized runtime must independently restore the exact parent Adam/RNG state,
validate execution parity, and own the guarded optimizer/checkpoint loop. This
module reuses the existing four backward implementations without changing any
target, weight, cache boundary or recorded-call behavior.
"""
import math

from evals.context_native_complete_batch import matches, validate_exposure
from evals.context_native_stream_training import exposure as stream_exposure
from evals.native_control_training import exposure as control_exposure
from evals.native_reasoning_training import exposure as reasoning_exposure

START = 528
UPDATES = 8
RATE = 1e-6
SCHEMA = 'dl-native-saved528-continuation-objective-v1'


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative measured loss required')
    return value


def _weighted(rows, field):
    if not rows:
        raise ValueError('An objective group cannot be empty')
    return math.fsum(_number(r['weight']) * _number(r[field]) for r in rows)


def loss_record(row):
    """Sum the actual backward coefficients, not an unweighted CE average.

    Supplemental partitions differ between updates. The sum is the sampled
    pre-update objective, not an all-case endpoint or convergence measurement.
    """
    retained = {name: _weighted(rows, 'loss')
                for name, rows in row['complete_balanced_microbatches'].items()}
    supplemental = dict(
        native49_text=_weighted(row['native_stream_teacher_microbatch'],
                                'balanced_spoken_clock_ce'))
    for name, field in (('control40', 'control40_microbatch'),
                        ('reasoning32', 'reasoning32_microbatch')):
        for kind in ('text', 'acoustic'):
            supplemental[name + '_' + kind] = _weighted(row[field], kind + '_ce')
    total = math.fsum([*retained.values(), *supplemental.values()])
    _number(total)
    return dict(schema=SCHEMA, stage_step=row['step'],
                before_global_adam_step=START + row['step'] - 1,
                planned_after_global_adam_step=START + row['step'],
                retained154=retained, supplemental=supplemental,
                sampled_pre_update_objective=total,
                fixed_complete_endpoint=False, convergence_established=False,
                behavior_qualified=False)


def backward(context, local_step):
    """Accumulate all existing objectives once; leave stepping to the runtime."""
    if type(local_step) is not int or not 0 <= local_step < UPDATES:
        raise ValueError('Only local steps0..7 of the new bounded stage are valid')
    sizes = dict(batches=28, retention=4, references=4, native_teacher_batches=25,
                 additional_batches=24, warm_references=10, clock_references=14,
                 audio49_batches=49, stream_bound=49, control_bound=40,
                 reasoning_bound=32)
    if any(len(context[name]) != size for name, size in sizes.items()):
        raise ValueError('Complete retained154/native49/control40/reasoning32 context required')
    if context['args'].actual_response_first_word_weight != 16:
        raise ValueError('The original first-word coefficient must remain16')
    from evals.context_native_complete_batch import backward as complete_backward
    from evals.context_native_stream_training import backward as stream_backward
    from evals.native_control_training import backward as control_backward
    from evals.native_reasoning_training import backward as reasoning_backward
    lm = context['lm']
    groups = complete_backward(lm, *(context[k] for k in (
        'batches', 'retention', 'references', 'native_teacher_batches',
        'additional_batches', 'component_kinds', 'response_boundaries', 'audits',
        'warm_references', 'clock_references', 'audio49_batches', 'forward')),
        context['args'].actual_response_first_word_weight)
    audio = context['audio49_policy']
    matches(groups, audio['complete_batch_membership'])
    row = dict(step=local_step + 1,
               global_adam_step=START + local_step + 1,
               complete_balanced_microbatches=groups,
               assistant_audio49_microbatch=groups['audio49'],
               additional_native_v2_microbatch=groups['new24'],
               native_stream_teacher_microbatch=stream_backward(
                   lm, context['stream_bound'], audio['native_stream_auxiliary'], local_step),
               control40_microbatch=control_backward(lm, context['control_bound'], local_step),
               reasoning32_microbatch=reasoning_backward(lm, context['reasoning_bound'], local_step))
    row['loss_record'] = loss_record(row)
    return row


def validate_stage(rows, audio_policy, control_policy, reasoning_policy):
    """A completed stage requires full exposure and real global Adam clocks."""
    if (len(rows) != UPDATES or [r['step'] for r in rows] != list(range(1, 9))
            or any(type(r['step']) is not int for r in rows)
            or [r['global_adam_step'] for r in rows] != list(range(529, 537))
            or any(type(r['global_adam_step']) is not int for r in rows)):
        raise ValueError('Exactly8 NEW updates529..536 required; never repeat520..528')
    validate_exposure(rows, audio_policy['teacher_ids'],
                      audio_policy['source_unique49_acoustic_targets'],
                      audio_policy['complete_batch_membership'])
    stream_exposure(rows, audio_policy['native_stream_auxiliary'])
    control_exposure(rows, control_policy)
    reasoning_exposure(rows, reasoning_policy)
    for row in rows:
        if (type(row['learning_rate']) not in (int, float) or row['learning_rate'] != RATE
                or row['all26_gradients_finite'] is not True
                or type(row['frozen_parameter_gradients']) is not int
                or row['frozen_parameter_gradients'] != 0
                or _number(row['pre_clip_gradient_norm']) <= 0
                or type(row['gradient_clip_norm']) not in (int, float)
                or row['gradient_clip_norm'] != 1.
                or row['loss_record'] != loss_record(row)):
            raise ValueError('Changed rate, gradient ownership or measured objective receipt')
        if any(type(row[key]) is not int or row[key] < 0
               for key in ('nonzero_A_gradients', 'nonzero_B_gradients')):
            raise ValueError('Measured nonnegative adapter gradient counts required')
    if any(not any(row[key] > 0 for row in rows)
           for key in ('nonzero_A_gradients', 'nonzero_B_gradients')):
        raise ValueError('Both A and B adapters must receive a live gradient in this stage')
    return dict(schema=SCHEMA, complete_exposure=True, new_optimizer_updates=8,
                initial_global_adam_step=START, final_global_adam_step=536,
                original_eight_updates_repeated=False,
                numerical_restore_approved=False, behavior_qualified=False,
                model_adopted=False)
