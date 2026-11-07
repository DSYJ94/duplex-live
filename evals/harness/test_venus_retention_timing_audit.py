import copy
import pytest
from data.venus.retention_timing_audit import verify_action


def test_empty_final_requires_actual_word_crossing_not_an_invented_extra_second():
    aligned=dict(text='surroundings.',source_sha256='original-audio',
                 measured_words=[dict(text='surroundings.',start=.4,end=1.16)])
    action=dict(text='surroundings.',timing_source_sha256='original-audio',frame=5,
                native_speech_units=[dict(offset=0,text='surroundings.',final=False),
                                     dict(offset=1,text='',final=True)])
    assert verify_action(action,aligned)['empty_final']
    changed=copy.deepcopy(aligned);changed['measured_words'][0]['end']=.9
    with pytest.raises(ValueError):verify_action(action,changed)
    changed=copy.deepcopy(action);changed['native_speech_units'][0]['final']=True
    with pytest.raises(ValueError):verify_action(changed,aligned)


def test_correct_looking_units_do_not_excuse_different_audio_or_words():
    aligned=dict(text='Done.',source_sha256='original-audio',
                 measured_words=[dict(text='Done.',start=.1,end=.9)])
    action=dict(text='Done.',timing_source_sha256='different-audio',frame=5,
                native_speech_units=[dict(offset=0,text='Done.',final=True)])
    with pytest.raises(ValueError):verify_action(action,aligned)
