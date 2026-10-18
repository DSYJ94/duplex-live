"""Evaluation-only ordinary/reasoning/capability retention, actual backend receipts."""
import argparse
import itertools
import json
from pathlib import Path
import time


def oracle(value):
    if not isinstance(value, dict):
        return value
    if value['oracle'] == 'subset':
        counts = {0: 1}
        for number in value['numbers']:
            new = counts.copy()
            for total, count in counts.items():
                new[total + number] = new.get(total + number, 0) + count
            counts = new
        return counts.get(value['target'], 0)
    if value['oracle'] == 'assignment':
        matrix = value['costs']
        return min(sum(matrix[i][j] for i, j in enumerate(order)) for order in itertools.permutations(range(len(matrix))))
    if value['oracle'] == 'congruences':
        import math
        return next(i for i in range(1, math.prod(p[0] for p in value['pairs']) + 1) if all(i % a == b for a, b in value['pairs']))
    raise ValueError('Unknown post-execution oracle')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('assets', 'kernel-assets', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--candidate', type=Path)
    a = ap.parse_args()
    from evals.moshi_model_download import sha256
    fixture = Path('evals/fixtures/context-generalization-20261018.json')
    plan = json.loads(fixture.read_text())
    if plan['training_approved'] or not plan['do_not_train'] or len(plan['cases']) != 24:
        raise ValueError('Requires fixed evaluation-only inputs')
    a.out.mkdir(exist_ok=False, parents=True)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor = Qwen35ResponseExecutor(a.assets, kernel_assets=a.kernel_assets, pending_speech_protocol='explicit-actions', compact_replies=True)
    warm_long_prefill(executor)
    from training.control.head_adapter import parameter_hashes
    source = dict(executor.model.named_parameters())
    before = parameter_hashes(source)
    candidate_proof = None
    if a.candidate:
        from training.control.context_adapter import install, restore
        proof = json.loads((a.candidate / 'restore-proof.json').read_text())
        trained = json.loads((a.candidate / 'training-proof.json').read_text())
        if (not proof['fresh_process_restore_passed'] or trained['optimizer_steps'] != 324 or trained['phase'] != 'full'
                or proof['training_proof_sha256'] != sha256(a.candidate / 'training-proof.json')
                or proof['adapter_weights_sha256'] != sha256(a.candidate / 'candidate-adapter/adapter.safetensors')):
            raise ValueError('Requires actual full candidate restore proof')
        adapters = install(executor.model)
        restore(a.candidate / 'candidate-adapter', adapters, sha256(a.assets))
        candidate_proof = dict(restore_proof_sha256=sha256(a.candidate / 'restore-proof.json'), adapter_weights_sha256=proof['adapter_weights_sha256'])
    from evals.duplex_reasoning_worker import ReasoningWorld, run_reasoning_worker, ask_reasoner
    from data.synthetic.filter import norm_words, _num2words
    attempts = 0
    rows = []
    for spec in plan['cases']:
        row = dict(id=spec['id'], kind=spec['kind'], question=spec['question'], completed=False)
        def backend(question):
            nonlocal attempts
            if spec['external_request_forbidden']:
                raise RuntimeError('User constraint forbids any actual external request')
            if attempts >= plan['actual_reasoning_cap_per_arm']:
                raise RuntimeError('Predeclared actual reasoning limit reached')
            attempts += 1
            return ask_reasoner(question)
        world = ReasoningWorld({}, backend)
        context = [dict(role='user', text=spec['question'])]
        revision = world.reserve('current', context)
        start, previous = time.monotonic(), attempts
        try:
            result = run_reasoning_worker(context, world, 'current', executor.model_id, revision, 'state-aware', [], executor)
            row.update(completed=True, result=result)
        except Exception as error:
            row.update(error_type=type(error).__name__, message=str(error), evidence=getattr(error, 'result', None))
        expected = oracle(spec['expected_for_review'])
        row.update(expected_after_execution=expected, oracle_supplied_to_models=False,
                   actual_remote_requests=attempts - previous, actual_world_receipts=world.receipts, seconds=time.monotonic()-start,
                   external_request_forbidden=spec['external_request_forbidden'], semantic_review_required=not isinstance(expected, int))
        if isinstance(expected, int):
            content = row.get('result', {}).get('content', '')
            row['number_only_passed'] = row['completed'] and norm_words(content) in (norm_words(_num2words(expected)), str(expected))
        rows.append(row)
        (a.out / (spec['id'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    if before != parameter_hashes(source):
        raise ValueError('Evaluation changed source parameters')
    report = dict(results=rows, cases=len(rows), completed=sum(r['completed'] for r in rows), actual_remote_requests=attempts,
                  load=executor.proof, candidate=candidate_proof, source_hashes_unchanged=True, fixture_sha256=sha256(fixture),
                  code_sha256=sha256(Path(__file__)), optimizer_steps=0, training_approved=False, all_rows_retained=True,
                  automatic_overall_qualification=False, speech_qualified=False, native_duplex_qualified=False, production_adopted=False)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cases=len(rows), completed=report['completed'], actual_remote_requests=attempts, semantic_review_pending=True)), flush=True)


if __name__ == '__main__':
    main()
