import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from evals.native_fresh_probe import validate_options,load,finish,observe,user_stream


class FreshProbe(unittest.TestCase):
    def options(self,**change):
        d=dict(train_history=False,backend_mode='observe-only',private_value=None,ret_logit_bias=0.,
            warm_audio=False,pending_status=False,adapter_disabled=False,adapter_zero=False,layout_audit=None,
            event_log_mode='immediate',backend_reference_profile='answer-only',persona_profile='recorded-records',
            math_attention=False,data=Path('data'),native_fresh_quality=Path('quality'),scenario='native-fresh-000',
            expected_kind='simple',lookup_delivery_delay_s=0.,native_fresh_behavior_review=Path('review'),native_fresh_candidate_proof=Path('candidate'),adapter_proof=Path('proof'))
        d.update(change);return SimpleNamespace(**d)

    def test_training_reference_or_diagnostic_options_fail_closed(self):
        validate_options(self.options())
        for field,value in [('train_history',True),('private_value','white heart'),('backend_mode','scheduled'),
                            ('ret_logit_bias',1.),('pending_status',True),('adapter_zero',True),
                            ('warm_audio',True),('math_attention',True),('layout_audit',Path('override'))]:
            with self.subTest(field=field),self.assertRaises(ValueError):validate_options(self.options(**{field:value}))

    def test_predeclared_backend_kind_and_delay_cannot_be_changed(self):
        safe=dict(kind='quiet',backend='observe-only',minimum_lookup_delay_s=0.,path='user.wav',wav_sha256='abc')
        with patch('evals.native_fresh_inputs.inference_input',return_value=safe):
            source,row=load(self.options());self.assertEqual(row['split'],'fresh-evaluation');self.assertEqual(row['kind'],'simple')
            for d in (dict(expected_kind='private'),dict(lookup_delivery_delay_s=1.),dict(backend_mode='native-asr')):
                with self.assertRaises(ValueError):load(self.options(**d))

    def test_v2_routes_to_its_own_full20_guard(self):
        safe=dict(kind='quiet',backend='observe-only',minimum_lookup_delay_s=0.,path='v2.wav',wav_sha256='abc')
        with patch('evals.native_fresh_inputs_v2.inference_input',return_value=safe) as v2, \
             patch('evals.native_fresh_inputs.inference_input') as old, \
             patch('evals.native_fresh_review.validate') as reviewed:
            source,row=load(self.options(scenario='native-fresh2-016'))
            v2.assert_called_once();old.assert_not_called();reviewed.assert_called_once_with(Path('review'),Path('candidate'),Path('proof'));self.assertEqual(row['path'],'v2.wav')

    def test_v2_does_not_append_a_second_undeclared_tail(self):
        import numpy as np
        from evals.native_fresh_inputs_v2 import PLAN_SHA256
        audio=np.zeros(24000,dtype=np.float32)
        self.assertIs(user_stream(audio,24000,dict(scenario='native-fresh2-000',plan_sha256=PLAN_SHA256)),audio)
        with self.assertRaises(ValueError):user_stream(audio,24000,dict(scenario='native-fresh2-000',plan_sha256='changed'))
        self.assertEqual(len(user_stream(audio,24000,dict(scenario='native-fresh-000'))),21*24000)

    def test_derived_profile_is_explicit_and_still_requires_candidate_review(self):
        safe=dict(kind='quiet',backend='observe-only',minimum_lookup_delay_s=0.,path='v2.wav',wav_sha256='abc')
        args=self.options(scenario='native-fresh2-016',native_fresh_source_profile='owner-approved-turbo-spelling-v1')
        with patch('evals.native_fresh_source_reapproval.inference_input',return_value=safe) as derived, \
             patch('evals.native_fresh_inputs_v2.inference_input') as original, \
             patch('evals.native_fresh_review.validate') as reviewed:
            load(args);derived.assert_called_once();original.assert_not_called();reviewed.assert_called_once()
            reviewed.side_effect=ValueError('Native behavior not reviewed')
            with self.assertRaises(ValueError):load(args)
            self.assertEqual(derived.call_count,1)
        with self.assertRaises(ValueError):
            load(self.options(native_fresh_source_profile='owner-approved-turbo-spelling-v1'))

    def test_output_is_evaluation_only_and_forced_or_training_results_are_rejected(self):
        base=dict(scheduled_dispatch=False,word_or_acoustic_tokens_forced=False,response_start_forced=False,
                  student_history=None,training_approved=False,real_user_samples_processed=1920,real_user_samples_expected=1920)
        with tempfile.TemporaryDirectory() as root:
            args=self.options(out=Path(root));p=args.out/'report.json';p.write_text(json.dumps(base));finish(args,{'scenario':'fresh'})
            result=json.loads(p.read_text());self.assertTrue(result['fresh_holdout']);self.assertFalse(result['full20_task_qualification_passed'])
            for field,value in [('scheduled_dispatch',True),('student_history',{'labels':'unwanted'}),
                                ('word_or_acoustic_tokens_forced',True),('real_user_samples_processed',100)]:
                p.write_text(json.dumps(dict(base,**{field:value})))
                with self.assertRaises(RuntimeError):finish(args,{})

    def test_observer_preserves_all_samples_and_unnecessary_native_requests(self):
        import numpy as np
        class Engine:
            frame_size=1920
            def step(self,chunk):return dict(text_token=3,agent_pcm=np.full(1920,.1,dtype=np.float32),is_ret=True)
            def decode_text(self,tokens):return 'native raw text'
        with tempfile.TemporaryDirectory() as root:
            args=self.options(out=Path(root),source_split='fresh-evaluation',scenario='native-fresh-000')
            with patch('torch.cuda.synchronize'),patch('evals.native_fresh_probe.time.sleep'):
                observe(Engine(),np.zeros(4000,dtype=np.float32),args,dict(fresh_evaluation_input={'scenario':'fresh'}))
            r=json.loads((args.out/'report.json').read_text());self.assertEqual(r['real_user_samples_processed'],4000)
            self.assertEqual(r['native_request_markers_observed_not_executed'],3)
            self.assertFalse(r['quality_qualified']);self.assertIsNone(r['student_history'])


if __name__=='__main__':unittest.main()
