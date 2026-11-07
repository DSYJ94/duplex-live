import pytest
from training.venus.native_product_review import case_review


def fixture():
    result=dict(id='original',family='ordinary',complete=True,input_frames=1,native_turn_still_open=False)
    speech=dict(id='original',word_pass=False,raw_peak_pass=True,references=dict(intended_user_facing='Correct words.'),asr_text='Wrong words.',user_facing_text_wer=.5)
    frames=[dict(sequence=0,audio_samples=24000,peak=0.,elapsed=.7,wall_since_start=.7,
                 speech_completion=dict(pending=False,events=[dict(kind='terminal_acoustic_piece',reason='acoustic_eos')]))]
    return result,speech,frames


def test_acoustic_eos_and_inaudible_waveform_do_not_waive_word_failure():
    row=case_review(*fixture())
    assert not row['word_pass'] and not row['speech_or_reasoning_success_claimed']
    assert row['no_voiced_frames_but_ASR_word_failure']
    assert row['terminal_acoustic_eos_events']==1 and row['voiced_frames']==0


def test_partial_trace_rejected():
    result,speech,frames=fixture();result['input_frames']=2
    with pytest.raises(ValueError):case_review(result,speech,frames)


def test_backlog_is_synthetic_processing_evidence_not_live_latency():
    result,speech,frames=fixture();frames[0].update(peak=.2,elapsed=2.2,wall_since_start=2.2)
    row=case_review(result,speech,frames)
    assert row['frames_over_one_second']==1
    assert row['maximum_synthetic_frame_backlog_seconds']==pytest.approx(1.2)
