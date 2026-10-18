"""One fixed synthetic API evaluation of actual terminal exact-count actions."""
import argparse
import itertools
import json
from pathlib import Path
from evals.exact_reasoner import ask_exact_reasoner
from evals.moshi_model_download import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    fixture = Path('evals/fixtures/exact-count-reasoner-plan-20261018.json')
    plan = json.loads(fixture.read_text())
    if plan['training_approved'] or not plan['do_not_train'] or len(plan['cases']) != plan['actual_http_request_cap']:
        raise ValueError('Exact fixed evaluation-only API budget required')
    a.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for spec in plan['cases']:
        row = dict(id=spec['id'], question=spec['question'], passed=False, api_attempted=True)
        try:
            result = ask_exact_reasoner(spec['question'])
            row['result'] = result
            receipts = result['exact_tool_receipts']
            checks = dict(one_actual_http_request=result['actual_http_requests'] == 1,
                          expected_tool_choice=bool(receipts) == spec['tool_expected'])
            if 'numbers' in spec:
                # A separate exhaustive algorithm, after actual model execution.
                numbers, target = spec['numbers'], spec['target']
                oracle = sum(sum(c) == target for size in range(len(numbers)+1) for c in itertools.combinations(numbers, size))
                row['post_execution_exhaustive_oracle'] = oracle
                if receipts:
                    checks.update(executed=receipts[0]['executed'], original_list_exact=receipts[0]['arguments']['numbers'] == numbers,
                                  original_target_exact=receipts[0]['arguments']['target'] == target, correct_count=receipts[0]['value'] == oracle,
                                  output_is_actual_receipt=result['content'] == receipts[0]['content'])
                else:
                    from data.synthetic.filter import norm_words
                    checks['correct_count'] = norm_words(result['content']) == norm_words(str(oracle))
            row.update(checks=checks, passed=all(checks.values()), semantic_review_required=spec.get('semantic_review_required', False))
        except Exception as error:
            row.update(error_type=type(error).__name__, message=str(error))
        rows.append(row)
        (a.out / (spec['id'] + '.json')).write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps(dict(id=row['id'], passed=row['passed'], error=row.get('message'))), flush=True)
    report = dict(cases=len(rows), mechanically_passed=sum(r['passed'] for r in rows), results=rows,
                  api_request_attempts_upper_bound=len(rows), confirmed_completed_http_requests=sum(r.get('result', {}).get('actual_http_requests', 0) for r in rows),
                  fixture_sha256=sha256(fixture), code_sha256=sha256(Path(__file__)),
                  teacher_or_eval_oracle_sent_to_model=False, optimizer_steps=0, semantic_review_pending=True,
                  owner_demo_changed=False, training_approved=False, production_adopted=False, native_duplex_qualified=False)
    (a.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
