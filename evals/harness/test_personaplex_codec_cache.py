"""Exercise the REAL shared encode helper's entire sidecar contract on CPU."""
import numpy as np
import torch
import pytest
from training.personaplex.codec_cache import native_sidecar
from evals.native_control_cache import encode_item


def test_real_encoding_entrypoint_accepts_full_ordinary_metadata(monkeypatch):
    original=torch.as_tensor
    def cpu_tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original(*args,**kwargs)
    monkeypatch.setattr(torch,'as_tensor',cpu_tensor)
    class Codec:
        def encode(self,audio):
            assert audio.shape==(2,1,3840)
            return torch.zeros((2,8,2),dtype=torch.int64)
    class Builder:
        def prepare_item(self,labels,duration):return torch.tensor([[[5,3]]])
    class Tokenizer:
        def vocab_size(self):return 32000
        def encode(self,text):assert text=='yes';return [5]
    row=dict(fold='TRAIN',id='venus-retention-request-v2-train-ordinary-0000')
    side=native_sidecar(row,dict(alignments=[['yes',[0.,.08],'SPEAKER_MAIN']]))
    codes,timing=encode_item(np.zeros((3840,2),dtype=np.float32),side,Codec(),Builder(),Tokenizer())
    assert codes.shape==(1,17,2) and timing['word_count']==1
    with pytest.raises(ValueError):native_sidecar(dict(row,fold='HOLDOUT'),dict(alignments=[]))
