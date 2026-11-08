import numpy as np
import pytest
from data.personaplex.source_export import pack


def test_stereo_keeps_assistant_left_user_right_and_every_sample():
    user=np.full(24000,.2,dtype=np.float32);answer=np.full(4800,-.3,dtype=np.float32)
    audio,labels,start=pack(user,answer,[dict(text=' yes',start=.02,end=.15)])
    assert start>=len(user)+.16*24000 and start%1920==0
    np.testing.assert_array_equal(audio[:len(user),1],user)
    np.testing.assert_array_equal(audio[start:start+len(answer),0],answer)
    assert not audio[:start,0].any() and not audio[len(user):,1].any()
    assert labels==[[' yes',[start/24000+.02,start/24000+.15],'SPEAKER_MAIN']]
    assert len(audio)%1920==0


@pytest.mark.parametrize('peak',[1.,float('nan')])
def test_bad_audio_is_rejected_without_gain_repair(peak):
    a=np.full(24000,peak,dtype=np.float32)
    with pytest.raises(ValueError,match='peak gate'):pack(a,np.zeros(24000,dtype=np.float32),[])


def test_alignment_cannot_overrun_or_invent_source_times():
    a=np.zeros(24000,dtype=np.float32)
    with pytest.raises(ValueError,match='fit source'):pack(a,a,[dict(text='word',start=.8,end=1.2)])
    with pytest.raises(ValueError,match='monotonic'):pack(a,a,[dict(text=' ',start=0,end=.1)])
