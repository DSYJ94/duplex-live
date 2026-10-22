"""Control40 cache contracts; fixtures do not qualify native codec execution."""
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from evals import native_control_cache as cache

class Tokenizer:
    def encode(self,text):return [7]
    def vocab_size(self):return 32000

class NativeControlCache(unittest.TestCase):
    def test_integer_grid_never_invents_a_final_frame(self):
        for n in range(1,20001):self.assertEqual(math.ceil(cache.frame_duration(n)*12.5),n)
        for n in (0,True,1.5):
            with self.assertRaises(ValueError):cache.frame_duration(n)

    def test_all40_five_families_train_only_are_required(self):
        rows=[dict(scenario_id=f'{family}-{i}',split='train',family=family,scenario_class='native_control3_teacher')
              for family in sorted(cache.FAMILIES) for i in range(8)]
        conversion=dict(accepted=rows,complete40=True,training_approved=False,channel_order=['assistant','user'])
        self.assertEqual(len(cache.inventory(conversion)),40)
        for bad in (dict(conversion,accepted=rows[:-1]),dict(conversion,channel_order=['user','assistant']),
                    dict(conversion,accepted=[dict(rows[0],split='test')]+rows[1:])):
            with self.assertRaises(ValueError):cache.inventory(bad)

    def test_failed_approval_prevents_cuda_initialization_or_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out'
            with patch.object(cache,'verify_conversion',side_effect=ValueError('Whole40 failed')),patch.object(cache,'runtime') as runtime:
                with self.assertRaisesRegex(ValueError,'Whole40 failed'):cache.build(root,root,root,root,out)
                runtime.assert_not_called()
            self.assertFalse(out.exists())

    def test_full_pcm_channel_ownership_and_no_acoustic_crop(self):
        frames=3;audio=np.stack([np.full(frames*1920,.1),np.full(frames*1920,.2)],axis=1).astype('float32')
        original=audio.copy();side=dict(reference_arrivals=[],timeline=dict(meta=dict(split='train'),events=[]),
                                        alignments=[['Fixture',[.08,.16],'SPEAKER_MAIN']])
        class Mimi:
            extra=0
            def encode(self,value):
                np.testing.assert_array_equal(value.numpy(),audio.T[:,None])
                result=torch.full((2,8,frames+self.extra),11,dtype=torch.int64);result[1]=22
                return result
        class Builder:
            def prepare_item(self,alignments,duration):
                self.expected=alignments
                result=torch.full((1,1,math.ceil(duration*12.5)),3,dtype=torch.int64);result[0,0,1]=7
                return result
        real_as_tensor=torch.as_tensor
        def cpu_fixture(*args,**kwargs):kwargs['device']='cpu';return real_as_tensor(*args,**kwargs)
        mimi=Mimi()
        with patch.object(torch,'as_tensor',side_effect=cpu_fixture):
            codes,timing=cache.encode_item(audio,side,mimi,Builder(),Tokenizer())
            self.assertTrue(torch.all(codes[:,1:9]==11));self.assertTrue(torch.all(codes[:,9:17]==22))
            self.assertEqual(timing['max_first_word_offset_ms'],0.)
            mimi.extra=1
            with self.assertRaisesRegex(ValueError,'every original PCM frame'):cache.encode_item(audio,side,mimi,Builder(),Tokenizer())
        np.testing.assert_array_equal(audio,original)
        codes[0,0,0]=4
        with self.assertRaisesRegex(ValueError,'spurious delegation'):cache.check_codes(codes,frames,side,Tokenizer())

if __name__=='__main__':unittest.main()
