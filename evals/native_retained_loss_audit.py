"""Read-only accounting of the failed native7590052 pilot's saved losses.

Step losses were computed before their optimizer update. All154 retained terms
have stable membership; the supplementary native49 schedule rotates examples.
Neither accounting nor a declining teacher-forced loss establishes causality,
convergence, free-running behavior, or permission to alter retention objectives.
"""
import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

from evals.context_native_complete_batch import validate_groups

REPORT_SHA = '397a259d124948dec3aeed63564ad24dbf2f13bb2b3699565f9f1974e60fa3eb'


def weighted(rows):
    values = [r['weight'] * r['loss'] for r in rows]
    if any(not math.isfinite(v) for v in values):
        raise ValueError('Finite recorded weighted losses required')
    result = math.fsum(values)
    independent = sum(Decimal(str(r['weight'])) * Decimal(str(r['loss'])) for r in rows)
    if abs(result - float(independent)) > 1e-12:
        raise ValueError('Independent decimal accounting differs')
    return result


def audit(report):
    if (report['updates_this_run'] != 8 or report['adam_parameter_steps_after'] != [528]
            or report['warm_start']['optimizer_steps'] != 520):
        raise ValueError('Exact eight-update native520 child required')
    steps = report['steps']
    if [s['step'] for s in steps] != list(range(1, 9)):
        raise ValueError('All eight original step records required')
    curves = []; identity = None; stream_ids = []
    for step in steps:
        groups = step['complete_balanced_microbatches']
        validate_groups(groups)
        current = {g: [(r['id'], r['weight']) for r in rows] for g, rows in groups.items()}
        if identity is not None and current != identity:
            raise ValueError('Retained comparison changed membership or weights')
        identity = current
        for name in ('reference', 'word', 'clock'):
            for row in groups[name]:
                if abs(row['loss'] - (row['kl'] + 10 * row['hidden_mse'])) > 1e-7:
                    raise ValueError('Recorded KL/hidden loss formula differs')
        stream = step['native_stream_teacher_microbatch']
        stream_ids.extend(r['id'] for r in stream)
        norm = step['gradient_norm_before_clip']
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('Actual finite positive gradient norm required')
        losses = {g: weighted(rows) for g, rows in groups.items()}
        curves.append(dict(logged_step=step['step'],
            optimizer_updates_already_applied_when_losses_computed=step['step'] - 1,
            retained154_weighted_group_losses=losses,
            retained154_weighted_sum=math.fsum(losses.values()),
            rotating_native_stream_weighted_loss=math.fsum(
                r['weight'] * r['balanced_spoken_clock_ce'] for r in stream),
            rotating_native_stream_ids=[r['id'] for r in stream],
            gradient_norm_before_clip=norm,
            approximate_unit_clip_multiplier=min(1., 1. / (norm + 1e-6))))
    if len(stream_ids) != 49 or len(set(stream_ids)) != 49:
        raise ValueError('Complete rotating49 exposure required')
    changes = {}
    for name, first in steps[0]['complete_balanced_microbatches'].items():
        last = steps[-1]['complete_balanced_microbatches'][name]
        deltas = [b['loss'] - a['loss'] for a, b in zip(first, last)]
        before = curves[0]['retained154_weighted_group_losses'][name]
        after = curves[-1]['retained154_weighted_group_losses'][name]
        changes[name] = dict(components=len(first), weight_mass=math.fsum(r['weight'] for r in first),
            before_first_update=before, before_eighth_update=after, weighted_delta=after-before,
            component_loss_increases=sum(v > 0 for v in deltas),
            component_loss_decreases=sum(v < 0 for v in deltas),
            component_loss_unchanged=sum(v == 0 for v in deltas))
    actual = steps[0]['complete_balanced_microbatches']['actual']
    metrics = []
    for field in ('initial_stage_metrics', 'final_metrics'):
        recorded = report[field]
        if len(recorded) != 28 or any(a['scenario'] != b['scenario'] for a, b in zip(actual, recorded)):
            raise ValueError('Exact ordered28 initial/final metrics required')
        values = [b.get('optimization_nll', b['nll']) for b in recorded]
        if field == 'initial_stage_metrics' and any(abs(v-a['loss']) > 1e-7 for a, v in zip(actual, values)):
            raise ValueError('Initial metric criterion differs from actual training loss')
        metrics.append(weighted([dict(weight=a['weight'], loss=v) for a, v in zip(actual, values)]))
    return dict(schema='dl-native7590052-retained-loss-accounting-v1',
        complete_step_records=8, retained_components_per_update=154,
        independently_recomputed_decimal_sums=True, stable_retained_membership_and_weights=True,
        curves=curves, group_changes=changes,
        retained154_weighted_sum_before_first_update=curves[0]['retained154_weighted_sum'],
        retained154_weighted_sum_before_eighth_update=curves[-1]['retained154_weighted_sum'],
        all_recorded_steps_clip_gradients=all(c['gradient_norm_before_clip'] > 1 for c in curves),
        actual28_initial_and_post_eighth_update=dict(before=metrics[0], after=metrics[1],
            weighted_delta=metrics[1]-metrics[0], scope='Same ordered28 actual-history metrics and objective weights; optimization_nll or unchanged pad-weighted nll for no-content targets'),
        comparison_scope='Before update1 versus before update8: zero versus seven new updates; not the complete post-update8 objective',
        rotating_stream_loss_is_same_examples_convergence_curve=False,
        held_out_convergence_measured=False, loss_group_gradient_directions_measured=False,
        complete_post_update8_objective_measured=False, causal_failure_component_identified=False,
        actual_API_calls=0, GPU_calls=0, optimizer_updates=0, model_or_loss_changes=False,
        training_or_extension_or_adoption_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    data = args.report.read_bytes()
    if hashlib.sha256(data).hexdigest() != REPORT_SHA:
        raise ValueError('Original native7590052 report hash required')
    result = audit(json.loads(data))
    result.update(source_report=str(args.report), source_report_sha256=REPORT_SHA,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.out.open('x') as writer:
        writer.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in (
        'retained154_weighted_sum_before_first_update',
        'retained154_weighted_sum_before_eighth_update',
        'all_recorded_steps_clip_gradients')}))


if __name__ == '__main__':
    main()
