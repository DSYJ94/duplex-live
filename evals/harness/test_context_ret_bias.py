import unittest
from types import SimpleNamespace
import torch
from evals.context_ret_bias import adjust,install,validate


class RetBiasTest(unittest.TestCase):
    def test_zero_policy_is_exact_identity(self):
        z=torch.randn(1,1,3,9)
        self.assertIs(adjust(z,0.),z)

    def test_only_ret_changes_and_original_logits_remain_untouched(self):
        z=torch.arange(27,dtype=torch.float32).reshape(1,1,3,9);original=z.clone()
        out=adjust(z,-2.)
        self.assertTrue(torch.equal(z,original))
        difference=out-z;self.assertTrue(torch.equal(difference[...,4],torch.full((1,1,3),-2.)))
        difference[...,4]=0;self.assertEqual(int(torch.count_nonzero(difference)),0)
        self.assertTrue(torch.equal(adjust(z,-2.),out))

    def test_hidden_and_other_graph_paths_are_preserved(self):
        h=torch.randn(1,1,4);z=torch.randn(1,1,1,9);inp=object()
        def original(value):self.assertIs(value,inp);return h,z
        direct=object();state=SimpleNamespace(graphed_main=original,graphed_embeddings=direct)
        engine=SimpleNamespace(lm_gen=SimpleNamespace(_streaming_state=state))
        metadata=install(engine,-4.);actual,biased=state.graphed_main(inp)
        self.assertIs(actual,h);self.assertIs(state.graphed_embeddings,direct)
        self.assertTrue(torch.equal(biased,adjust(z,-4.)))
        self.assertEqual(engine._dl_ret_bias_calls,1)
        self.assertFalse(metadata['scenario_dependent']);self.assertFalse(metadata['text_or_audio_tokens_forced'])
        with self.assertRaises(RuntimeError):install(engine,-2.)

    def test_invalid_bias_or_nontext_layout_is_rejected(self):
        for v in (float('nan'),float('inf'),-9.,.1):
            with self.subTest(value=v),self.assertRaises(ValueError):validate(v)
        with self.assertRaises(RuntimeError):adjust(torch.zeros(1,16,1,9),-2.)


if __name__=='__main__':unittest.main()
