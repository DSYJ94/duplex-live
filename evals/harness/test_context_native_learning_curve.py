"""Fixture-only checks for read-only lineage and approved target masks."""
import copy
import unittest
from evals.context_native_learning_curve import PINS,DATA_SHA256,validate_lineage,target_metrics


class NativeLearningCurve(unittest.TestCase):
    def setUp(self):
        self.parent=dict(optimizer_steps=520,data_sha256=DATA_SHA256,frozen_hashes={'base':'fixed'},
            voice_sha256='voice',assets_manifest_sha256='assets')
        self.children=[]
        for pin in PINS:
            identity=dict(checkpoint_sha256=pin['checkpoint'],training_state_sha256=pin['state'],
                reload_report_sha256=pin['reload'],report_sha256=pin['report'])
            child=dict(self.parent,optimizer_steps=pin['steps'],repo_commit=pin['source'],passed=True,
                warm_start=dict(checkpoint_sha256='parent'),additional_native_v2=dict(complete24_per8=True,learning_rate=2e-6),
                warm_ordinary_retention=dict(weight=64),warm_ordinary_clock_retention=dict(weight=64),
                retention_teacher_text=dict(approved='all25'),native_protocol=dict(name='fixed'))
            self.children.append((identity,child))
        self.children[1][1].update(initial_optimizer_steps=8,updates_this_run=32,
            resumed_from_training_state='/runs/context-training-proof-7556562/training-state.pt',
            source_transition=dict(training_state_sha256=PINS[0]['state'],from_repo_commit=PINS[0]['source'],to_repo_commit=PINS[1]['source']))

    def check(self,children=None):
        validate_lineage(self.parent,self.children if children is None else children,'parent',DATA_SHA256)

    def test_exact_declared_learning_curve_lineage(self):self.check()

    def test_omitted_changed_state_or_data_is_rejected(self):
        with self.assertRaises(ValueError):self.check(self.children[:1])
        for field in ['checkpoint_sha256','training_state_sha256','reload_report_sha256','report_sha256']:
            bad=copy.deepcopy(self.children);bad[1][0][field]='wrong'
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(bad)
        for field,value in [('data_sha256','changed'),('voice_sha256','changed'),('frozen_hashes',{}),
                ('warm_ordinary_retention',dict(weight=0)),('warm_ordinary_clock_retention',dict(weight=0))]:
            bad=copy.deepcopy(self.children);bad[1][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(bad)

    def test_changed_continuation_or_teacher_policy_is_rejected(self):
        for field,value in [('initial_optimizer_steps',0),('updates_this_run',40),
                ('resumed_from_training_state','/runs/wrong/training-state.pt'),('retention_teacher_text',{}),
                ('source_transition',dict(training_state_sha256='wrong',from_repo_commit=PINS[0]['source'],to_repo_commit=PINS[1]['source']))]:
            bad=copy.deepcopy(self.children);bad[1][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.check(bad)

    def test_only_approved_targets_count_in_word_and_silence_measurements(self):
        import torch
        z=torch.zeros(1,1,3,8);z[0,0,0]=float('nan');z[0,0,1,6]=8.;z[0,0,2,3]=8.
        mask=torch.tensor([False,True,True]);targets=torch.tensor([999,6,3])
        r=target_metrics(z,mask,targets)
        self.assertEqual((r['frames'],r['spoken_frames'],r['pad3_frames']),(2,1,1))
        self.assertEqual((r['spoken_top1_matches'],r['pad3_top1_matches']),(1,1))
        self.assertGreater(r['first_spoken_target_probability'],.99)

    def test_pure_silence_is_valid_and_nonfinite_supervised_values_are_rejected(self):
        import torch
        z=torch.zeros(1,1,2,8);mask=torch.tensor([True,True]);target=torch.tensor([3,3])
        r=target_metrics(z,mask,target);self.assertEqual(r['spoken_frames'],0)
        self.assertNotIn('first_spoken_target_probability',r)
        z[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):target_metrics(z,mask,target)


if __name__=='__main__':unittest.main()
