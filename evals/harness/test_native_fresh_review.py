"""Fixture mutation checks for full native review before any second-corpus output."""
import copy
import json
import unittest
from unittest.mock import patch
from evals.harness import test_context_native_extension_review as fixtures
from evals.native_fresh_review import validate_review,validate
from evals.native_fresh_inputs_v2 import PLAN_SHA256


class NativeFreshReviewTests(unittest.TestCase):
    setUp = fixtures.NativeExtensionReviewTests.setUp

    def fresh(self):
        r=copy.deepcopy(self.review)
        r.update(schema='dl-native-v2-development-review-v1',fresh_plan_sha256=PLAN_SHA256,
                 fresh_evaluation_authorized=True,checkpoint_adopted=False)
        return r

    def check(self,r):return validate_review(r,self.proof,self.parent,self.identity)

    def test_complete_review_authorizes_measurement_without_adoption(self):
        self.assertFalse(self.check(self.fresh())['checkpoint_adopted'])

    def test_wrong_candidate_plan_restore_or_partial_review_cannot_qualify(self):
        for field,value in [('candidate_job','101'),('fresh_plan_sha256','other'),
                            ('independent_reload_report_sha256','other'),('all10_native_cases_reviewed',False),
                            ('all4_ordinary_core_correct',False),('fresh_evaluation_authorized',False),('checkpoint_adopted',True)]:
            r=self.fresh();r[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(r)
        r=self.fresh();r['cases'].pop()
        with self.assertRaises(ValueError):self.check(r)

    def test_review_cannot_hide_changed_audio_or_unreviewed_private_case(self):
        r=self.fresh();r['cases'][9]['behavior_and_receipts_reviewed']=False
        with self.assertRaises(ValueError):self.check(r)
        (self.runs/'native-v2-development-200_7/model.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.check(self.fresh())

    def test_locked_candidate_review_also_guards_baseline_and_rejects_other_adapter(self):
        with patch('evals.context_rollout_conditioning.verify_checkpoint',return_value=(self.identity,self.parent)), \
             patch('evals.native_fresh_review.committed_document',return_value=self.fresh()):
            self.assertTrue(validate('review',self.proof,self.proof.parent/'context-training-proof-7510331')['fresh_evaluation_authorized'])
            with self.assertRaises(ValueError):validate('review',self.proof,self.proof.parent/'context-training-proof-999')
        with self.assertRaises(ValueError):validate(None,self.proof,self.proof)

    def test_rejected_legacy_pilot_cannot_reach_fresh_outputs(self):
        self.parent['additional_native_v2']['complete24_per8']=False
        with self.assertRaises(ValueError):self.check(self.fresh())

    def test_rehashed_diagnostic_runtime_cannot_silently_become_stock_fresh_runtime(self):
        from evals.moshi_model_download import sha256
        for diagnostic in ('native_sampling_diagnostic','runtime_precision_diagnostic'):
            review=self.fresh();p=self.runs/'native-v2-development-200_0/report.json';r=json.loads(p.read_text())
            r['loading'].pop('native_sampling_diagnostic',None);r['loading']['adapter'].pop('runtime_precision_diagnostic',None)
            target=r['loading'] if diagnostic=='native_sampling_diagnostic' else r['loading']['adapter']
            target[diagnostic]=dict(diagnostic_only=True)
            p.write_text(json.dumps(r));review['cases'][0]['run_report_sha256']=sha256(p)
            with self.subTest(diagnostic=diagnostic),self.assertRaises(ValueError):self.check(review)

    def test_same_physical_run_alias_is_valid_but_another_directory_is_not(self):
        from evals.moshi_model_download import sha256
        alias=self.runs/'runs-alias';alias.symlink_to(self.runs,target_is_directory=True)
        audit=json.loads(self.speech.read_text())
        for index,row in enumerate(audit['results']):row['run']=str(alias/('native-v2-development-200_'+str(index)))
        self.speech.write_text(json.dumps(audit));review=self.fresh();review['independent_speech_report_sha256']=sha256(self.speech)
        self.assertTrue(self.check(review)['fresh_evaluation_authorized'])
        other=self.runs/'other-run';other.mkdir();(other/'model.wav').write_bytes((self.runs/'native-v2-development-200_0/model.wav').read_bytes())
        audit['results'][0]['run']=str(other);self.speech.write_text(json.dumps(audit));review['independent_speech_report_sha256']=sha256(self.speech)
        with self.assertRaises(ValueError):self.check(review)


if __name__=='__main__':unittest.main()
