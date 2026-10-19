import unittest
from types import SimpleNamespace
from evals.native_text_sampling_diagnostic import policy,case,verify_engine


class NativeTextSamplingTests(unittest.TestCase):
    def args(self):
        return SimpleNamespace(native_sampling_diagnostic=True,native_text_temperature=0.,backend_mode='observe-only',
            expected_kind='simple',train_history=True,adapter_proof='proof',adapter_disabled=False,adapter_zero=False,
            warm_audio=False,ret_logit_bias=0.,pending_status=False,persona_profile='recorded-records',
            lookup_delivery_delay_s=0.,scenario='recovery-train-012')

    def test_default_unchanged_and_nondefault_requires_optin(self):
        a=self.args();a.native_sampling_diagnostic=False;a.native_text_temperature=.7
        self.assertIsNone(policy(a));a.native_text_temperature=0.
        with self.assertRaises(ValueError):policy(a)

    def test_explicit_diagnostic_cannot_change_persona_gain_or_enable_worker_training(self):
        a=self.args();p=policy(a);self.assertFalse(p['native_quality_or_extension_approved'])
        for key,value in [('backend_mode','native-asr'),('warm_audio',True),('ret_logit_bias',1.),
                ('adapter_zero',True),('persona_profile','released-default'),('scenario','native-fresh2-000')]:
            a=self.args();setattr(a,key,value)
            with self.assertRaises(ValueError):policy(a)
        verify_engine(SimpleNamespace(lm_gen=SimpleNamespace(temp_text=0.,temp=.8,use_sampling=True)),p)
        with self.assertRaises(RuntimeError):verify_engine(SimpleNamespace(lm_gen=SimpleNamespace(temp_text=.7,temp=.8,use_sampling=True)),p)

    def test_all8_pairs_fixed_and_never_training_approval(self):
        rows=[case(i,'7556562') for i in range(8)]
        self.assertEqual([r['checkpoint_job'] for r in rows],['7510331','7556562']*4)
        self.assertEqual(len({r['scenario'] for r in rows}),4)
        self.assertTrue(all(not r['training_approved'] for r in rows))
        for i in [-1,8]:
            with self.assertRaises(ValueError):case(i,'7556562')

    def test_low_nonzero_temperature_explicit_and_consistent(self):
        a=self.args();a.native_text_temperature=.3;p=policy(a)
        self.assertEqual(p['text_temperature'],.3);self.assertEqual(case(1,'7556562',.3)['text_temperature'],.3)
        verify_engine(SimpleNamespace(lm_gen=SimpleNamespace(temp_text=.3,temp=.8,use_sampling=True)),p)
        a.native_text_temperature=.5
        with self.assertRaises(ValueError):policy(a)
        with self.assertRaises(ValueError):case(1,'7556562',.5)

    def test_upstream_zero_temperature_uses_argmax_without_text_rng_draw(self):
        import torch
        from contextspan.moshi.utils.sampling import sample_token
        logits=torch.tensor([[0.,2.,1.],[4.,1.,-1.]])
        before=torch.get_rng_state().clone();chosen=sample_token(logits,use_sampling=True,temp=0.,top_k=250)
        self.assertTrue(torch.equal(chosen,logits.argmax(-1)));self.assertTrue(torch.equal(before,torch.get_rng_state()))


if __name__=='__main__':unittest.main()
