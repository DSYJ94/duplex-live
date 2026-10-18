import os
from pathlib import Path
import sys
import unittest


class StreamingClockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder=os.environ.get('KYUTAI_TTS_COMPAT_FOLDER')
        if folder:
            from evals.kyutai_tts_compat import verified_overlay
            _,overlay=verified_overlay(Path(folder));sys.path.insert(0,str(overlay))

    def model(self):
        import torch
        from moshi.models.lm import LMModel,LMGen
        lm=LMModel(n_q=32,dep_q=0,card=16,text_card=8,dim=16,num_heads=2,
            num_layers=1,delays=[0]*33,causal=True,context=32,
            norm='rms_norm_f32',gating='silu',device='cpu',dtype=torch.float32).eval()
        return lm,LMGen(lm,temp=0.,temp_text=0.)

    def test_first_actual_frame_is_rejected_without_initial_lm_bootstrap(self):
        import torch
        from moshi.utils.compile import no_compile
        from evals.kyutai_stt_probe import checked_step
        _,gen=self.model()
        with torch.no_grad(),no_compile(),gen.streaming(1):
            with self.assertRaisesRegex(RuntimeError,'replaced before'):
                checked_step(gen,torch.ones(1,32,1,dtype=torch.int64))

    def test_every_actual_frame_reaches_forward_after_bootstrap_and_reset(self):
        import torch
        from moshi.utils.compile import no_compile
        from evals.kyutai_stt_probe import bootstrap,checked_step
        _,gen=self.model()
        with torch.no_grad(),no_compile(),gen.streaming(1):
            for _ in range(2):
                gen.reset_streaming();bootstrap(gen)
                for value in (1,2,3,4,5):
                    checked_step(gen,torch.full((1,32,1),value,dtype=torch.int64))
                with self.assertRaises(RuntimeError):bootstrap(gen)


if __name__=='__main__':unittest.main()
