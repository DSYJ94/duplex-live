import json
from fractions import Fraction
from data.venus.reasoning_curriculum import build,synthetic,messages,read_jsonl,normalized
from data.venus.reasoning_pool import arithmetic


def test_every_synthetic_target_is_exact_and_questions_are_distinct():
    rows=[synthetic(k,n) for k in range(16) for n in list(range(32))+[100,101]]
    assert len(rows)==len({normalized(r['question']) for r in rows})==544
    for r in rows:assert arithmetic(r['verification']['expression'])==Fraction(r['numeric_target'])


def test_jsonl_reader_preserves_unicode_line_separators_inside_strings(tmp_path):
    row={'question':'There are two colours.\u2028How many colours?','answer':'2'}
    path=tmp_path/'pool.jsonl';path.write_text(json.dumps(row,ensure_ascii=False)+'\n')
    assert read_jsonl(path)==[row]


def test_complete_curriculum_keeps_public_train_provenance_and_references_out_of_inputs():
    pool=[dict(id='public'+str(i),source_row=i,source_split='train',all_present_annotations_exact=True,question='Unique public question '+str(i),original_solution='The result is 2.\n#### 2',numeric_target='2') for i in range(550)]
    corpus=build(pool,[])
    assert sum(r['fold']=='TRAIN' for r in corpus['rows'])==1008
    assert sum(r['fold']=='DEV' for r in corpus['rows'])==64
    assert len({normalized(r['question']) for r in corpus['rows']})==1072
    for row in corpus['rows']:
        changed=dict(row,response='unrelated',numeric_target='999')
        assert messages(corpus,row)==messages(corpus,changed)
        assert messages(corpus,row,answer=True)[-1]['content']==row['response']
        if row['source_kind']=='gsm8k_official_train':assert row['original_solution'].endswith('#### 2')
