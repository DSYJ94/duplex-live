import hashlib,json
from pathlib import Path
import pytest
from evals.venus_reasoning16_plan import build,messages
from training.venus.foundation_reasoning import numeric_score,reused_reports,PRIOR_SOURCE
from training.venus.reasoning128 import PLAN_SHA,prompt
from evals.venus_reasoning128_plan import build as mcplan


def test_numeric_controls_have_checked_answers_and_no_reference_in_prompt():
    plan=build()
    assert len(plan['cases'])==16
    assert [r['expected'] for r in plan['cases']]==list(map(str,[22,17,225,31,32,15,31,24,20,28,39,19,5,15,7,4]))
    for r in plan['cases']:
        assert messages(plan,r)==messages(plan,dict(r,expected='impossible',reference_derivation='private'))
    assert json.loads(Path('evals/fixtures/venus-numeric16-text-20261108.json').read_text())==plan
    questions={r['question'] for r in mcplan()['cases']}
    for f in ('venus-control-corpus-v4-20261026.json','venus-control-repair-source-v2-20261027.json','venus-retention-request-source-proposal-v2-20261107.json'):
        for s in json.loads((Path('evals/fixtures')/f).read_text())['scenes']:
            if s['fold']=='TRAIN':questions.update(s['user_turns'])
    assert not questions&{r['question'] for r in plan['cases']}


@pytest.mark.parametrize('raw,expected,correct,valid',[
    ('22','22',True,True),('The answer is 22.','22',True,True),('2.2e1','22',True,True),
    ('23','22',False,True),('22 or 23','22',False,False),('No answer','22',False,False),
    ('22.5','22',False,True),('3 - 1 = 2','2',False,False)])
def test_numeric_score_does_not_accept_substrings_or_multiple_alternatives(raw,expected,correct,valid):
    assert numeric_score(raw,expected)==dict(correct=correct,single_numeric_result=valid)


def test_reused_results_preserve_original_source_and_bind_every_actual_prompt(tmp_path):
    plan=mcplan();inputs={'reused_files':{}}
    rows=[dict(id=r['id'],input_sha256=hashlib.sha256(prompt(plan,r).encode()).hexdigest()) for r in plan['cases']]
    for name in ('parent646.json','expanded-data.json','rehearsal.json','comparison.json'):
        p=tmp_path/name;p.write_text(json.dumps(dict(source=PRIOR_SOURCE,plan_sha256=PLAN_SHA,complete128=True,cases=rows)))
        inputs['reused_files'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(reused_reports(tmp_path,inputs,plan,None))==3
    changed=json.loads((tmp_path/'expanded-data.json').read_text());changed['cases'][3]['input_sha256']='wrong'
    p=tmp_path/'expanded-data.json';p.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='hash differs'):reused_reports(tmp_path,inputs,plan,None)
    inputs['reused_files']['expanded-data.json']=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='prompts differ'):reused_reports(tmp_path,inputs,plan,None)
