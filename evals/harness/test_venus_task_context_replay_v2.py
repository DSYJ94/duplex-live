import hashlib
from training.venus.task_context_replay_v2 import replay_frames


def frame(i,text,events=()):
    return dict(sequence=i,raw_text=text,events=list(events),input_sha256=hashlib.sha256(bytes(32000)).hexdigest())


def event(query,ident='actual'):
    return dict(type='native_delegate',work_id=ident,query=query,raw_query=query)


def test_nested_opening_binds_actual_latest_opening_not_earlier_or_closing_audio():
    frames=[frame(0,'<delegate>lookup'),frame(1,'<delegate>lookup lantern'),frame(2,'</delegate>',[event('lookup lantern')])]
    rows,broken=replay_frames(frames,bytes(32000*3))
    assert len(rows)==1 and rows[0]['evidence']['last_sequence']==1
    assert rows[0]['evidence']['after_opening_audio_included'] is False
    assert broken==[dict(kind='nested_opening_prior_span_abandoned',abandoned_opening_sequence=0,new_opening_sequence=1)]


def test_two_real_dispatches_and_unclosed_tail_all_preserved():
    frames=[frame(0,'<delegate>lookup lantern</delegate><delegate>lookup tray</delegate>',[event('lookup lantern','one'),event('lookup tray','two')]),frame(1,'<delegate>unfinished')]
    rows,broken=replay_frames(frames,bytes(32000*2))
    assert [r['work_id'] for r in rows]==['one','two'] and len(broken)==1
    assert broken[0]['kind']=='unclosed_final_span_not_dispatched'


def test_disagreeing_actual_dispatch_is_rejected():
    import pytest
    with pytest.raises(ValueError):replay_frames([frame(0,'<delegate>lookup tray</delegate>',[event('lookup lantern')])],bytes(32000))
