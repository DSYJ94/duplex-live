import copy
import unittest
from evals.context_engine_replay import collector_identity


class EngineCollectorTest(unittest.TestCase):
    def setUp(self):
        self.zero={'adapter':{'zero':True,'disabled':False},'layout':{'checked':True},'input_warmup':None}

    def test_original_zero_collector_remains_explicit_default(self):
        self.assertIsNone(collector_identity(self.zero))

    def test_trained_weights_cannot_silently_enter_zero_replay(self):
        self.zero['adapter']['zero']=False
        with self.assertRaises(RuntimeError):collector_identity(self.zero)

    def test_explicit_trained_mode_does_not_fall_back_to_zero(self):
        with self.assertRaises(RuntimeError):collector_identity(self.zero,True)

    def test_warmed_disabled_or_calibrated_collectors_are_rejected(self):
        for change in ('warm','disabled','calibrated'):
            with self.subTest(change=change):
                loading=copy.deepcopy(self.zero);loading['adapter']['zero']=False
                if change=='warm':loading['input_warmup']={'frames':16}
                if change=='disabled':loading['adapter']['disabled']=True
                if change=='calibrated':loading['ret_logit_calibration']={'logit_bias':-2.}
                with self.assertRaises(RuntimeError):collector_identity(loading,True)


if __name__=='__main__':unittest.main()
