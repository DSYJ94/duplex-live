"""Fixture mutation checks for full native review before any second-corpus output."""
import copy
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


if __name__=='__main__':unittest.main()
