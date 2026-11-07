from types import SimpleNamespace
from data.venus.retention_semantics import calibration,calibration_v2
from training.venus.native_schedule import schedule


def engine():
    return SimpleNamespace(tokenizer=SimpleNamespace(encode=lambda text,**kw:list(range(100,100+len(text.split())))),
        speak_token_id=1,listen_token_id=2,turn_eos_token_id=3,chunk_eos_token_id=4,chunk_tts_eos_token_id=5)


def test_original_positive_control_declares_four_units_but_schedules_only_one():
    row=calibration()[0];action=row['payload']['actions'][0]
    actual=schedule(engine(),dict(actions=[action],frames=20))
    assert set(actual)=={3}
    assert set(actual)!=set(range(*action['output_interval_seconds']))


def test_v2_control_units_match_declared_native_intervals_without_changing_answers():
    old,new=calibration(),calibration_v2()
    assert len(new)==16 and sum(r['expected'] for r in new)==8
    assert not {r['id'] for r in old}&{r['id'] for r in new}
    for before,after in zip(old,new):
        assert before['expected']==after['expected']
        assert before['payload']['user']==after['payload']['user']
        assert before['payload']['actual_feedback']==after['payload']['actual_feedback']
        for original,action in zip(before['payload']['actions'],after['payload']['actions']):
            for key in ('frame','kind','text','final'):assert action[key]==original[key]
            if action['kind']!='speak':continue
            units=action['native_speech_units']
            assert ' '.join(u['text'] for u in units)==action['text']
            actual=schedule(engine(),dict(actions=[action],frames=20))
            assert set(actual)==set(range(*action['output_interval_seconds']))
        if after['expected']:
            schedule(engine(),dict(actions=after['payload']['actions'],frames=20))
    # No mutation of the original fixtures or invented recorded word timings.
    assert old==calibration()
    assert all('native_speech_units' not in a for r in old for a in r['payload']['actions'])
