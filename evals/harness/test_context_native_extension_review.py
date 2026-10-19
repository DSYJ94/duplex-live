"""Fixture-only evidence binding, never proof of native behavior."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from evals.context_native_extension_review import committed_document, validate_review, validate_cases
from evals.moshi_model_download import sha256


class NativeExtensionReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.runs=Path(self.temp.name);self.proof=self.runs/'context-training-proof-100'
        self.proof.mkdir();self.identity=dict(checkpoint_sha256='adapter',training_state_sha256='state',reload_report_sha256='reload')
        self.parent=dict(optimizer_steps=8,additional_native_v2=dict(complete24_per8=True),voice_sha256='voice')
        plan=Path(__file__).parents[1]/'fixtures/native-v2-development-20261019.json'
        cases=json.loads(plan.read_text())['cases'];reviews=[];speech=[]
        for index,case in enumerate(cases):
            folder=self.runs/('native-v2-development-200_'+str(index));folder.mkdir()
            wav=folder/'model.wav';wav.write_bytes(('fixture-only-wave'+str(index)).encode())
            adapter=dict(checkpoint_sha256='adapter',voice_codes_sha256='voice',zero=False,disabled=False,installed_before_native_graphs=True)
            report=dict(audio_sha256=sha256(wav),loading=dict(adapter=adapter),scenario=case['scenario'],scheduled_dispatch=False,
                word_or_acoustic_tokens_forced=False,response_start_forced=False,real_user_samples_processed=100,real_user_samples_expected=100)
            p=folder/'report.json';p.write_text(json.dumps(report))
            reviews.append(dict(index=index,run_report_sha256=sha256(p),audio_sha256=sha256(wav),behavior_and_receipts_reviewed=True,
                speech_quality_issue_reviewed=True,reason='Fixture review of complete task',behavior_verdict='pass',ordinary_core_correct=True))
            speech.append(dict(run=str(folder),wav_sha256=sha256(wav),asr_time_bounds_valid=True,asr_word_bounds_valid=True,
                word_starts_within_recorded_audio=True,wer_against_model_text=0.))
        folder=self.runs/'native-v2-development-asr-300';folder.mkdir();self.speech=folder/'report.json'
        self.speech.write_text(json.dumps(dict(results=speech)))
        self.review=dict(schema='dl-native-v2-bounded-extension-review-v1',candidate_job='100',candidate_checkpoint_sha256='adapter',
            candidate_training_state_sha256='state',independent_reload_report_sha256='reload',evaluation_plan_sha256=sha256(plan),
            all10_native_cases_reviewed=True,all4_ordinary_core_correct=True,bounded_extension_authorized=True,candidate_adopted=False,
            native_candidate_rejected_for_extension=False,extension_optimizer_updates=32,native_evaluation_array_job='200',speech_audit_job='300',
            independent_speech_report_sha256=sha256(self.speech),cases=reviews)

    def check(self,review=None,parent=None):
        return validate_review(self.review if review is None else review,self.proof,self.parent if parent is None else parent,self.identity)

    def test_complete_review_binds_all10_without_implying_adoption(self):
        self.assertFalse(self.check()['candidate_adopted'])

    def test_subset_wrong_candidate_or_unreviewed_task_rejected(self):
        for field in ('subset','candidate','reviewed','extension','adoption'):
            bad=copy.deepcopy(self.review)
            if field=='subset':bad['cases'].pop()
            elif field=='candidate':bad['candidate_job']='101'
            elif field=='reviewed':bad['cases'][6]['behavior_and_receipts_reviewed']=False
            elif field=='extension':bad['bounded_extension_authorized']=False
            else:bad['candidate_adopted']=True
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(bad)

    def test_changed_raw_audio_cannot_reuse_review(self):
        (self.runs/'native-v2-development-200_9/model.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.check()

    def test_changed_or_failed_ordinary_independent_speech_rejected(self):
        original=json.loads(self.speech.read_text())
        for field,value in [('asr_time_bounds_valid',False),('wer_against_model_text',.1)]:
            bad=copy.deepcopy(original);bad['results'][0][field]=value;self.speech.write_text(json.dumps(bad))
            review=copy.deepcopy(self.review);review['independent_speech_report_sha256']=sha256(self.speech)
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(review)

    def test_rejected_legacy_candidate_or_failed_ordinary_core_cannot_extend(self):
        bad=copy.deepcopy(self.parent);bad['additional_native_v2']['complete24_per8']=False
        with self.assertRaises(ValueError):self.check(parent=bad)
        review=copy.deepcopy(self.review);review['cases'][0]['ordinary_core_correct']=False
        with self.assertRaises(ValueError):self.check(review)

    def test_actual_receipt_task_failure_can_only_receive_explicit_research_review(self):
        review=copy.deepcopy(self.review);review['cases'][6].update(behavior_verdict='fail',reason='Actual receipt reviewed; native result use still fails')
        self.assertFalse(self.check(review)['candidate_adopted'])

    def test_explicit_failed_pilot_evidence_binding_never_weakens_default_gate(self):
        review=copy.deepcopy(self.review);review['all4_ordinary_core_correct']=False
        review['cases'][0].update(ordinary_core_correct=False,behavior_verdict='fail')
        with self.assertRaises(ValueError):self.check(review)
        # Only the separate owner validator requests evidence-only binding.
        self.assertFalse(validate_cases(review,self.proof,self.parent,self.identity,
            require_ordinary=False)['cases'][0]['ordinary_core_correct'])
        (self.runs/'native-v2-development-200_9/model.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            validate_cases(review,self.proof,self.parent,self.identity,require_ordinary=False)


class CommittedReviewTests(unittest.TestCase):
    def test_changed_or_untracked_review_checkout_is_not_approval(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);p=root/'review.json';p.write_text('{}')
            def git(*args):
                return subprocess.check_output(['git','-C',str(root),'-c','user.name=Fixture',
                    '-c','user.email=fixture@example.invalid','-c','commit.gpgsign=false','-c','core.hooksPath=/dev/null',*args],stderr=subprocess.STDOUT)
            git('init','--quiet');git('add','review.json');git('commit','--quiet','-m','Fixture only')
            self.assertEqual(committed_document(p),{})
            p.write_text('{"changed":true}')
            with self.assertRaises(ValueError):committed_document(p)
            p.write_text('{}');(root/'untracked').write_text('fixture')
            with self.assertRaises(ValueError):committed_document(p)


if __name__=='__main__':unittest.main()
