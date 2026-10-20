import unittest
from pathlib import Path
from types import SimpleNamespace
from evals.native_seen_retention_diagnostic import case
from evals.native_text_sampling_diagnostic import policy,verify_engine


class SeenRetentionDiagnostic(unittest.TestCase):
    def args(self):
        return SimpleNamespace(native_seen_retention_diagnostic=True,native_sampling_diagnostic=False,
            native_text_temperature=.7,backend_mode='observe-only',expected_kind='simple',train_history=True,
            adapter_proof=Path('/projects/u6si/duplex-live/runs/context-training-proof-7558824'),
            adapter_disabled=False,adapter_zero=False,warm_audio=False,ret_logit_bias=0.,pending_status=False,
            persona_profile='recorded-records',lookup_delivery_delay_s=0.,scenario='recovery-train-008',
            voice='accepted-synthetic',data=Path('/projects/u6si/duplex-live/runs/recovery-data-7407554'))

    def test_all12_cases_keep_every_checkpoint_on_every_seen_train_question(self):
        rows=[case(i) for i in range(12)]
        self.assertEqual(len({(r['scenario'],r['checkpoint_job']) for r in rows}),12)
        self.assertTrue(all(r['diagnostic_outputs_must_not_be_trained_on'] for r in rows))
        for i in [-1,12,True]:
            with self.assertRaises(ValueError):case(i)

    def test_default_sampler_is_verified_and_capture_remains_unapproved(self):
        p=policy(self.args());verify_engine(SimpleNamespace(lm_gen=SimpleNamespace(temp_text=.7,temp=.8,use_sampling=True)),p)
        self.assertFalse(p['native_quality_or_extension_approved'])
        self.assertTrue(p['diagnostic_outputs_must_not_be_trained_on'])

    def test_no_profile_scene_checkpoint_voice_or_data_substitution(self):
        for field,value in [('native_sampling_diagnostic',True),('native_text_temperature',.3),
                ('backend_mode','native-asr'),('scenario','recovery-train-012'),('warm_audio',True),
                ('voice','released'),('persona_profile','released-default'),('adapter_zero',True),
                ('data',Path('/other/data')),('adapter_proof',Path('/runs/context-training-proof-7558824'))]:
            a=self.args();setattr(a,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):policy(a)

    def test_old_default_and_old_diagnostic_guard_are_unchanged(self):
        a=self.args();a.native_seen_retention_diagnostic=False
        self.assertIsNone(policy(a))
        a.native_sampling_diagnostic=True
        with self.assertRaises(ValueError):policy(a)


if __name__=='__main__':unittest.main()
