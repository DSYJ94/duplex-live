"""Synthetic authorization checks; no fixture here is owner or model approval."""
import copy
import unittest
from evals.context_native_exploration import (validate_authorization,
    PARENT_JOB, PARENT_SOURCE, CHECKPOINT, STATE, RELOAD, REVIEW, PROPOSAL)


class NativeExplorationAuthorization(unittest.TestCase):
    def setUp(self):
        self.source='a'*40
        self.identity=dict(checkpoint_sha256=CHECKPOINT,training_state_sha256=STATE,reload_report_sha256=RELOAD)
        self.parent=dict(repo_commit=PARENT_SOURCE,optimizer_steps=8,
            additional_native_v2=dict(complete24_per8=True,learning_rate=2e-6),
            warm_ordinary_retention=dict(weight=64),warm_ordinary_clock_retention=dict(weight=64))
        self.doc=dict(schema='dl-native-bounded32-owner-authorization-v1',
            operation='one-training-only-clock8-plus32',explicit_owner_selection=True,
            parent_job=PARENT_JOB,parent_source=PARENT_SOURCE,execution_source=self.source,
            checkpoint_sha256=CHECKPOINT,training_state_sha256=STATE,independent_reload_report_sha256=RELOAD,
            failed_complete10_review_sha256=REVIEW,proposal_sha256=PROPOSAL,
            starting_optimizer_steps=8,additional_optimizer_updates=32,final_optimizer_steps=40,
            optimizer_reset=False,learning_rate=2e-6,maximum_gpus=1,maximum_gpu_walltime_minutes=30,
            reservation='interactive',all_data_losses_voice_targets_masks_unchanged=True,
            failed_early_ordinary_and_speech_gate_acknowledged=True,
            full_native10_fresh20_and_adoption_gates_unchanged=True,
            candidate_adoption_authorized=False,further_training_after40_authorized=False,
            owner_reply='SYNTHETIC TEST ONLY - NOT OWNER APPROVAL',at_utc='fixture')

    def check(self,doc=None,identity=None,parent=None,source=None):
        return validate_authorization(self.doc if doc is None else doc,
            self.identity if identity is None else identity,self.parent if parent is None else parent,
            self.source if source is None else source)

    def test_exact_bounded_fixture_does_not_authorize_adoption_or_another_extension(self):
        result=self.check()
        self.assertFalse(result['candidate_adoption_authorized'])
        self.assertFalse(result['further_training_after40_authorized'])

    def test_missing_owner_decision_or_acknowledgment_cannot_enable_exploration(self):
        for field,value in [('owner_reply',''),('explicit_owner_selection',False),
                ('failed_early_ordinary_and_speech_gate_acknowledged',False),
                ('full_native10_fresh20_and_adoption_gates_unchanged',False)]:
            bad=copy.deepcopy(self.doc);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(doc=bad)

    def test_other_state_parent_source_or_anchor_is_rejected(self):
        for field in self.identity:
            bad=copy.deepcopy(self.identity);bad[field]='f'*64
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(identity=bad)
        for field,value in [('repo_commit','b'*40),('optimizer_steps',40),
                ('warm_ordinary_retention',dict(weight=0)),('warm_ordinary_clock_retention',dict(weight=0)),
                ('additional_native_v2',dict(complete24_per8=False,learning_rate=2e-6))]:
            bad=copy.deepcopy(self.parent);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(parent=bad)
        with self.assertRaises(ValueError):self.check(source='b'*40)

    def test_budget_reset_loss_or_adoption_scope_cannot_be_expanded(self):
        for field,value in [('additional_optimizer_updates',64),('final_optimizer_steps',72),
                ('optimizer_reset',True),('learning_rate',1e-5),('maximum_gpus',2),
                ('maximum_gpu_walltime_minutes',60),('all_data_losses_voice_targets_masks_unchanged',False),
                ('candidate_adoption_authorized',True),('further_training_after40_authorized',True),
                ('failed_complete10_review_sha256','f'*64),('proposal_sha256','f'*64)]:
            bad=copy.deepcopy(self.doc);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(doc=bad)


if __name__=='__main__':unittest.main()
