"""Matched inference-only schema experiment; no rewritten output or optimizer."""
import argparse
import json
from pathlib import Path
import time


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def fresh_controls(executor, folder):
    from training.control.paired_curriculum import payload, assess
    fixture = Path('evals/fixtures/context-fresh-controls-20261018.json')
    rows = []
    folder.mkdir()
    for scene in json.loads(fixture.read_text())['cases']:
        row = dict(id=scene['id'], scene=scene, passed=False)
        try:
            request, _, _, _ = payload(scene, executor.model_id)
            response = executor(request)
            quality = assess(scene, response)
            row.update(response=response, quality=quality, passed=quality['passed'])
        except Exception as error:
            row.update(error_type=type(error).__name__, message=str(error), evidence=getattr(error, 'result', None))
        rows.append(row)
        write(folder / (scene['id'] + '.json'), row)
    write(folder / 'report.json', dict(cases=len(rows), passed=sum(r['passed'] for r in rows), results=rows))
    return dict(cases=len(rows), passed=sum(r['passed'] for r in rows))


def local_retention(executor, folder):
    from training.control.paired_curriculum import payload
    from evals.context_generalization import oracle
    from data.synthetic.filter import norm_words
    rows = []
    folder.mkdir()
    for spec in json.loads(Path('evals/fixtures/context-generalization-20261018.json').read_text())['cases']:
        scene = dict(kind='delegate' if spec['kind'] == 'hard-reasoning' else 'ordinary', context=[dict(role='user', text=spec['question'])],
                     active_calculations=[], pending_speech=[], resume_target=None)
        row = dict(id=spec['id'], kind=spec['kind'], question=spec['question'], completed=False)
        try:
            request, _, _, _ = payload(scene, executor.model_id)
            response = executor(request)
            call = response['output'][0]
            arguments = json.loads(call['arguments'])
            row.update(completed=True, response=response, action=call['name'], arguments=arguments)
        except Exception as error:
            row.update(error_type=type(error).__name__, message=str(error), evidence=getattr(error, 'result', None))
        if spec['kind'] == 'hard-reasoning':
            row['selection_only_passed'] = row.get('action') == 'delegate_reasoning' and row.get('arguments') == {}
        else:
            expected = oracle(spec['expected_for_review'])
            row.update(expected_after_execution=expected, semantic_review_required=not isinstance(expected, int))
            if isinstance(expected, int):
                row['number_only_passed'] = row.get('action') == 'respond_to_user' and norm_words(row['arguments'].get('content', '')) == norm_words(str(expected))
        rows.append(row)
        write(folder / (spec['id'] + '.json'), row)
    summary = dict(cases=len(rows), completed=sum(r['completed'] for r in rows),
                   ordinary_numbers_passed=sum(r.get('number_only_passed', False) for r in rows),
                   hard_selection_only_passed=sum(r.get('selection_only_passed', False) for r in rows),
                   actual_reasoning_requests=0, semantic_review_pending=True)
    write(folder / 'report.json', dict(summary=summary, results=rows))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('assets', 'kernel-assets', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor = Qwen35ResponseExecutor(a.assets, kernel_assets=a.kernel_assets, pending_speech_protocol='explicit-actions', compact_replies=True)
    warm_long_prefill(executor)
    candidate = None
    if a.candidate:
        from evals.context_candidate import attach
        candidate = attach(executor, a.candidate, a.assets, 972)
    from training.control.head_adapter import parameter_hashes
    from training.control.paired_curriculum import specifications
    from training.control.context_evaluate import evaluate, retention
    from evals.nonempty_speech_contract import NonemptySpeechGenerator
    from evals.moshi_model_download import sha256
    held = [s for s in specifications() if s['split'] == 'heldout']
    before = parameter_hashes(dict(executor.model.named_parameters()))
    stages = {}
    for stage in ('original', 'nonempty'):
        if stage == 'nonempty':
            executor.generator = NonemptySpeechGenerator(executor.generator)
        folder = a.out / stage
        folder.mkdir()
        start = time.monotonic()
        controls = evaluate(executor, held, folder / 'controls')
        fresh = fresh_controls(executor, folder / 'fresh22')
        local = local_retention(executor, folder / 'local24')
        stages[stage] = dict(controls=controls, fresh=fresh, local=local, seconds=time.monotonic()-start)
        print(json.dumps(dict(stage=stage, summary=stages[stage])), flush=True)
    source_unchanged = before == parameter_hashes(dict(executor.model.named_parameters()))
    if not source_unchanged:
        raise ValueError('Inference-only experiment changed a model parameter')
    kept = retention(a.out / 'original/controls', a.out / 'nonempty/controls')
    requests = executor.generator.requests
    write(a.out / 'constraint-requests.json', requests)
    write(a.out / 'report.json', dict(stages=stages, control_retention=kept, candidate=candidate,
          all_parameters_unchanged=True, optimizer_steps=0, generated_outputs_repaired=False,
          actual_reasoning_requests=0, hard_questions_are_selection_only=True, all_failures_retained=True,
          semantic_review_pending=True, source_code_sha256=sha256(Path(__file__)),
          schema_adapter_sha256=sha256(Path('evals/nonempty_speech_contract.py')),
          production_adopted=False, speech_qualified=False, native_duplex_qualified=False))


if __name__ == '__main__':
    main()
