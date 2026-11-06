from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from portable.venus.wav_io import loader

@pytest.mark.parametrize('subtype',['FLOAT','PCM_16'])
def test_exact_samples_channels_and_frame_selection(tmp_path,subtype):
    path=tmp_path/'reference.wav'
    original=np.array([[0.,.25],[-.5,.75],[.125,-.25],[.5,0.]],dtype=np.float32)
    sf.write(path,original,16000,subtype=subtype)
    expected,sr=sf.read(path,dtype='float32',always_2d=True)
    load=loader(SimpleNamespace(from_numpy=lambda x:x))
    decoded,rate=load(path);assert rate==sr and np.array_equal(decoded,expected.T)
    selected,rate=load(path,frame_offset=1,num_frames=2,channels_first=False)
    assert np.array_equal(selected,expected[1:3])

def test_no_gain_reduction_or_clipping_concealment(tmp_path):
    path=tmp_path/'clipped.wav';sf.write(path,np.array([0.,1.1,-1.2],dtype=np.float32),16000,subtype='FLOAT')
    decoded,_=loader(SimpleNamespace(from_numpy=lambda x:x))(path)
    assert decoded.max()>1 and decoded.min() < -1

def test_unsupported_normalization_is_rejected(tmp_path):
    with pytest.raises(ValueError):loader(SimpleNamespace())(tmp_path/'unused.wav',normalize=False)
