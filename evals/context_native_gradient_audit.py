"""Complete TRAIN gradient diagnosis, without optimizer steps or model adoption."""
import json
import math
from pathlib import Path

from evals.moshi_model_download import sha256


def schedules(component_kinds, old_kinds, new_kinds):
    from evals.context_component_kind import balanced_index, retention_index
    from evals.context_native_teacher_objective import teacher_index
    result = {}
    for length in (8, 32):
        counts = {name: [0]*size for name, size in
                  [('actual', len(component_kinds)), ('reference', 4),
                   ('old25', len(old_kinds)), ('new24', len(new_kinds))]}
        for step in range(length):
            ri = retention_index(component_kinds, step, 4, True)
            counts['actual'][balanced_index(component_kinds, step)] += 1
            counts['reference'][ri] += 1
            counts['old25'][teacher_index(component_kinds, step, ri, old_kinds)] += 1
            counts['new24'][teacher_index(component_kinds, step, ri, new_kinds)] += 1
        assert all(sum(v) == length for v in counts.values())
        result[str(length)] = counts
    return result


def cosine(a, b):
    import torch
    denominator = float(a.double().norm()*b.double().norm())
    return None if denominator == 0. else float(torch.dot(a.double(), b.double())/denominator)


def run(lm, learned, actual, retention, references, old, new, kinds, boundaries,
        audits, args, metadata):
    import torch
    from torch.nn import functional as F
    from evals.personaplex_adapters import checkpoint_language_layers
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_teacher_text_balance import loss as teacher_loss
    from evals.context_decision_loss import weights as decision_weights
    if (len(actual) != 28 or len(old) != 25 or len(new) != 24 or len(retention) != 4
            or args.resume_training or args.reload or not args.warm_start_proof
            or args.actual_retention_teacher_weight != 1. or args.actual_base_retention_weight != 1.):
        raise ValueError('Gradient audit requires all28/25/24 and unchanged old retention at native520')
    before = tensor_hashes(lm.named_parameters())
    plan = schedules(kinds, [b[3]['kind'] for b in old], [b[3]['kind'] for b in new])
    width = sum(p.numel() for p in learned.values())
    means = {length: {name: torch.zeros(width, dtype=torch.float32) for name in counts}
             for length, counts in plan.items()}
    rows = []

    def measure(group, index, identity, make_loss):
        for p in learned.values():
            p.grad = None
        with checkpoint_language_layers(lm):
            loss = make_loss()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite diagnostic objective')
            loss.backward()
        gradients = []
        for p in learned.values():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise RuntimeError('Missing/nonfinite diagnostic gradient')
            gradients.append(p.grad.detach().float().flatten().cpu())
        vector = torch.cat(gradients)
        norm = float(vector.double().norm())
        if not math.isfinite(norm):
            raise RuntimeError('Nonfinite gradient norm')
        for length, counts in plan.items():
            means[length][group].add_(vector, alpha=counts[group][index]/int(length))
        row = dict(group=group, index=index, identity=identity, objective_loss=float(loss.detach()),
                   gradient_norm=norm, optimizer_steps=0)
        rows.append(row)
        print(json.dumps(row), flush=True)

    for index, (inputs, mask, target) in enumerate(actual):
        def loss(inputs=inputs, mask=mask, target=target, index=index):
            z = lm.forward_codes(inputs)[1][0, 0, mask].float(); y = target[mask]
            weights = decision_weights(y, args.actual_response_first_word_weight, boundaries[index])
            return (F.cross_entropy(z, y, reduction='none')*weights).sum()/weights.sum()
        measure('actual', index, audits[index]['scenario'], loss)
    for index, (inputs, mask, record) in enumerate(retention):
        def loss(inputs=inputs, mask=mask, index=index):
            h, z = lm.forward_codes(inputs); base_h, prob = references[index]
            return (F.kl_div(z[0, 0, mask].float().log_softmax(-1), prob, reduction='batchmean')
                    + 10.*(h[:, mask].float()-base_h.float()).square().mean())
        measure('reference', index, record['id'], loss)
    for group, batches in [('old25', old), ('new24', new)]:
        for index, (inputs, mask, target, record) in enumerate(batches):
            def loss(inputs=inputs, mask=mask, target=target):
                return teacher_loss(lm.forward_codes(inputs)[1][0, 0], target, mask)[0]
            measure(group, index, record['id'], loss)
    comparisons = {}
    for length, groups in means.items():
        pairs = {a+'__'+b: cosine(groups[a], groups[b]) for a in groups for b in groups if a < b}
        combined = sum(groups.values())
        comparisons[length] = dict(
            objective_mean_norms={name: float(value.double().norm()) for name, value in groups.items()},
            pairwise_cosine=pairs, combined_gradient_norm=float(combined.double().norm()),
            combined_cosine={name: cosine(value, combined) for name, value in groups.items()},
            interpretation='Negative cosine means opposing first-order directions at the fixed parent; not free-running behavior or Adam-update attribution')
    for p in learned.values():
        p.grad = None
    if before != tensor_hashes(lm.named_parameters()):
        raise RuntimeError('Zero-update gradient audit changed model weights')
    report = dict(schema='dl-native-v2-complete-gradient-diagnostic-v1',
        metadata=metadata, parent=str(args.warm_start_proof),
        parent_report_sha256=sha256(args.warm_start_proof/'report.json'),
        parent_checkpoint_sha256=sha256(args.warm_start_proof/'adapters.safetensors'),
        source_code_sha256=sha256(Path(__file__)), actual_cases=28, reference_cases=4,
        old_teacher_cases=25, new_teacher_cases=24, complete_gradient_cases=len(rows),
        all_model_weights_unchanged=True, optimizer_steps=0, new_checkpoint_written=False,
        native_behavior_evidence=False, training_approved=False, candidate_adopted=False,
        diagnostic_only=True, frozen_parent_gradients_not_sequential_optimizer_execution=True,
        schedule_counts=plan, comparisons=comparisons, cases=rows)
    assert len(rows) == 81
    (args.out/'native-v2-gradient-audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(complete_gradient_cases=81, optimizer_steps=0, comparisons=comparisons)), flush=True)
