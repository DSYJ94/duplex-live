import pytest
from portable.venus.task_context import CausalAudioBuffer


def test_opening_snapshot_cannot_acquire_later_audio_or_mutate():
    b=CausalAudioBuffer();raw=bytearray(b'a'*32000);b.append(0,raw)
    snapshot=b.freeze_at_request_opening();raw[0]=0
    b.append(1,b'b'*32000)
    assert snapshot.pcm16==b'a'*32000
    assert snapshot.receipt()['end_sample']==16000
    assert b.freeze_at_request_opening().receipt()['end_sample']==32000


def test_history_bound_keeps_original_absolute_clock():
    b=CausalAudioBuffer(2)
    for i in range(4):b.append(i,bytes([i])*32000)
    s=b.freeze_at_request_opening()
    assert (s.first_sequence,s.last_sequence)==(2,3)
    assert s.pcm16==bytes([2])*32000+bytes([3])*32000
    assert s.receipt()['start_sample']==32000 and s.receipt()['end_sample']==64000


def test_rejects_gaps_unbounded_history_and_missing_evidence():
    b=CausalAudioBuffer()
    with pytest.raises(ValueError):b.freeze_at_request_opening()
    with pytest.raises(ValueError):b.append(1,b'a'*32000)
    with pytest.raises(ValueError):b.append(0,b'a')
    with pytest.raises(ValueError):CausalAudioBuffer(31)
