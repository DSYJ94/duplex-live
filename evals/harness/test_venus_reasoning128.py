from collections import Counter
import json
from pathlib import Path
from evals.venus_reasoning128_plan import build,messages
from training.venus.reasoning128 import prompt,PLAN_SHA
import hashlib


def test_complete_frozen_cohort_and_reference_answers_stay_out_of_messages():
    plan=build()
    assert len(plan['cases'])==128
    assert Counter(x['category'] for x in plan['cases'])==dict(everyday=32,technical=24,context=24,ambiguity=16,delegation=24,stale=8)
    for row in plan['cases']:
        assert row['expected'] in 'ABCD' and len(set(row['options']))==4
        original=messages(plan,row);changed=dict(row,expected='D' if row['expected']!='D' else 'A')
        assert original==messages(plan,changed) and prompt(plan,row)==prompt(plan,changed)
        assert all(set(m)=={'role','content'} for m in original)
    path=Path('evals/fixtures/venus-reasoning128-text-20261107.json')
    assert json.loads(path.read_text())==plan
    assert hashlib.sha256(path.read_bytes()).hexdigest()==PLAN_SHA


def test_questions_are_not_exact_copies_of_training_user_turns():
    import re
    normalize=lambda s:re.sub(r'[^a-z0-9 ]','',s.lower()).strip()
    trained=set()
    for path in ('evals/fixtures/venus-control-repair-source-v2-20261027.json','evals/fixtures/venus-control-corpus-v4-20261026.json','evals/fixtures/venus-retention-request-source-proposal-v2-20261107.json'):
        for row in json.loads(Path(path).read_text())['scenes']:
            if row['fold']=='TRAIN':trained.update(normalize(x) for x in row['user_turns'])
    assert not trained&{normalize(r['question']) for r in build()['cases']}
