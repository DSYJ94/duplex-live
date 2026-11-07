from data.venus.retention_semantics import calibration,calibration_passed,verdict
import argparse,json


def test_always_accepting_or_invalid_reviewer_cannot_approve_independent_data_review():
    controls=calibration()
    assert not calibration_passed([dict(r,parsed={'passed':True,'reason':'Always yes'}) for r in controls])
    assert not calibration_passed([dict(r,parsed=None) for r in controls])
    assert calibration_passed([dict(r,parsed={'passed':r['expected'],'reason':'Fixture correct'}) for r in controls])
    assert all(set(r['payload'])=={'user','actions','actual_feedback'} for r in controls)


def test_review_parser_never_coerces_a_string_or_discards_unexplained_response_text():
    assert verdict('{"passed":true,"reason":"Correct"}')['passed'] is True
    assert verdict('{"passed":"true","reason":"Correct"}') is None
    assert verdict('Probably fine {"passed":true,"reason":"Correct"}') is None
    assert verdict('{"passed":true,"reason":"Correct","extra":"ignored"}') is None


def test_failed_calibration_never_reviews_or_approves_any_training_teacher(tmp_path):
    from data.venus.retention_semantics import collect_review
    visited=[]
    def judge(row):
        visited.append(row['id']);return dict(id=row['id'],parsed={'passed':True,'reason':'Always accepts'})
    report=collect_review(argparse.Namespace(output=tmp_path,source='test'),
        [{'id':'never-send-teacher','payload':{}}],'manifest',judge,'fixture')
    assert len(visited)==16 and 'never-send-teacher' not in visited
    assert not report['all96_semantic_pass'] and not report['complete96']
    assert not report['training_approved']
    assert not json.loads((tmp_path/'calibration.json').read_text())['all16_pass']


def test_qwen_final_content_requires_closed_thinking_and_never_retains_it():
    from data.venus.retention_semantics_qwen import final_content
    class Tokenizer:
        def decode(self,ids,**kw):
            assert ids==[8,9];return '{"passed":true,"reason":"Correct"}'
    raw,info=final_content([4,5,151668,8,9],Tokenizer())
    assert verdict(raw)['passed'] and info['thinking_tokens']==3
    assert final_content([4,5],Tokenizer())[0]==''


def test_public_reviewer_pin_requires_upstream_hash_for_every_weight_shard():
    from data.venus.reviewer_model import PIN,check_pin
    import pytest
    pin=json.loads(PIN.read_text());check_pin(pin)
    next(r for r in pin['files'] if r['path'].endswith('.safetensors'))['sha256']=None
    with pytest.raises(ValueError,match='upstream SHA256'):check_pin(pin)
