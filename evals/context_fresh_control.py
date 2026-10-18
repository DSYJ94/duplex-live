"""Unseen evaluation-only controls, separate from the reused126 development cases."""
import argparse
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('assets', 'kernel-assets', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--expected-updates', type=int, default=972)
    a = p.parse_args()
    from evals.moshi_model_download import sha256
    fixture = Path('evals/fixtures/context-fresh-controls-20261018.json')
    plan = json.loads(fixture.read_text())
    if not plan['do_not_train'] or plan['training_approved'] or len(plan['cases']) != 22:
        raise ValueError('Fixed evaluation-only controls required')
    a.out.mkdir(parents=True, exist_ok=False)
    from evals.qwen35_controller import Qwen35ResponseExecutor
    from evals.qwen35_prefill_warmup import warm_long_prefill
    executor = Qwen35ResponseExecutor(a.assets, kernel_assets=a.kernel_assets, pending_speech_protocol='explicit-actions', compact_replies=True)
    warm_long_prefill(executor)
    from training.control.head_adapter import parameter_hashes
    source = dict(executor.model.named_parameters())
    before = parameter_hashes(source)
    candidate = None
    if a.candidate:
        from evals.context_candidate import attach
        candidate = attach(executor, a.candidate, a.assets, a.expected_updates)
    from training.control.paired_curriculum import payload, assess
    rows = []
    for scene in plan['cases']:
        row = dict(id=scene['id'], kind=scene['kind'], scene=scene, passed=False)
        start = time.monotonic()
        try:
            request, _, _, _ = payload(scene, executor.model_id)
            response = executor(request)
            checked = assess(scene, response)
            row.update(response=response, quality=checked, passed=checked['passed'])
        except Exception as error:
            row.update(error_type=type(error).__name__, message=str(error), evidence=getattr(error, 'result', None))
        row['seconds'] = time.monotonic() - start
        rows.append(row)
        (a.out / (scene['id'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
    if before != parameter_hashes(source):
        raise ValueError('Fresh evaluation altered pretrained weights')
    report = dict(cases=len(rows), passed=sum(r['passed'] for r in rows), results=rows, source_load=executor.proof,
                  candidate=candidate, fixture_sha256=sha256(fixture), code_sha256=sha256(Path(__file__)),
                  source_hashes_unchanged=True, actual_reasoning_requests=0, delegation_is_selection_only=True,
                  cancellation_semantic_review_required=True, simulated_speech_state=True, physical_speech_executed=False,
                  training_approved=False, optimizer_steps=0, native_duplex_qualified=False, production_adopted=False)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(cases=len(rows), passed=report['passed'], semantic_review_required=True)), flush=True)


if __name__ == '__main__':
    main()
