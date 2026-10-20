import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from evals.context_native_audio49_training import validate_configuration,indices,validate_exposure

class Audio49PilotPolicy(unittest.TestCase):
    def args(self):
        return SimpleNamespace(native_audio49_audit=Path('/independent-audit'),native_audio49_weight=1.,
            native_audio49_preflight_only=False,native_audio49_preflight=Path('/independent-bf16-preflight'),
            actual_temporal_fit=True,native_v2_full24_per8=True,actual_fit_updates=8,resume_training=None,
            actual_mixed_data=list(range(28)),native_v2_learning_rate=2e-6,warm_start_proof=Path('/runs/context-training-proof-7510331'),
            actual_base_retention_weight=1.,actual_retention_teacher_weight=1.,actual_native_teacher_coverage='all-approved',
            actual_native_teacher_loss='balanced-spoken',actual_sampling='balanced-kind',actual_retention_sampling='balanced-kind',
            native_v2_warm_ordinary_weight=64.,native_v2_warm_ordinary_clock=True,actual_response_first_word_weight=16.,
            reload=False,native_v2_gradient_audit=False,native_v2_retention_drift_proof=None,native_v2_learning_curve_audit=False,
            native_v2_exploration_authorization=None)
    def test_parent_adam_is_an_explicit_new_smallstep_mode_only(self):
        a=self.args();a.native_audio49_parent_adam=True
        with self.assertRaises(ValueError):validate_configuration(a)
        a.native_audio49_step_scale=.1;self.assertTrue(validate_configuration(a))
        a.native_audio49_gradient_probe=True
        with self.assertRaises(ValueError):validate_configuration(a)
    def test_complete_batch_keeps_explicit_parent_moments_and_cannot_extend(self):
        a=self.args();a.native_audio49_complete_batch=True
        with self.assertRaises(ValueError):validate_configuration(a)
        a.native_audio49_parent_adam=True;a.native_audio49_step_scale=.1
        self.assertTrue(validate_configuration(a))
        a.actual_fit_updates=32
        with self.assertRaises(ValueError):validate_configuration(a)
    def test_new_pilot_requires_independent_audit_and_preserves_original_objectives(self):
        self.assertTrue(validate_configuration(self.args()))
        for field,value in [('actual_base_retention_weight',0.),('actual_retention_teacher_weight',0.),
                ('native_v2_warm_ordinary_weight',0.),('native_v2_warm_ordinary_clock',False),('actual_mixed_data',list(range(27))),
                ('actual_native_teacher_coverage','four-longest'),('actual_native_teacher_loss','legacy')]:
            a=self.args();setattr(a,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_configuration(a)
    def test_cannot_resume_rejected_joint_or40_or_silently_extend(self):
        for field,value in [('resume_training',Path('/runs/rejected/training-state.pt')),('actual_fit_updates',32),
                ('warm_start_proof',Path('/runs/context-training-proof-7558824')),('native_v2_learning_rate',1e-5)]:
            a=self.args();setattr(a,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_configuration(a)
    def test_smoke_cannot_optimize_or_masquerade_as_restore(self):
        a=self.args();a.native_audio49_preflight_only=True
        with self.assertRaises(ValueError):validate_configuration(a)
        a.native_audio49_preflight=None;self.assertTrue(validate_configuration(a))
        a.reload=True
        with self.assertRaises(ValueError):validate_configuration(a)
    def test_disabled_old_recipes_stay_disabled(self):
        a=SimpleNamespace();self.assertFalse(validate_configuration(a))
        a.native_audio49_weight=1.
        with self.assertRaises(ValueError):validate_configuration(a)
    def test_complete49_once_and_epoch_mean_weight_one(self):
        seen=[i for step in range(8) for i in indices(step)]
        self.assertEqual(sorted(seen),list(range(49)));self.assertEqual(len(set(seen)),49)
        self.assertAlmostEqual(sum(8/49 for i in seen)/8,1.)
        for bad in (-1,8,True,None):
            with self.assertRaises(ValueError):indices(bad)
    def steps(self):
        ids=['teacher-'+str(i) for i in range(49)]
        rows=[dict(step=s+1,assistant_audio49_microbatch=[dict(id=ids[i],weight=8/49,
            objective=dict(assistant_audio_targets=[2]*8,user_audio_loss_targets=0)) for i in indices(s)]) for s in range(8)]
        return ids,rows
    def test_complete_schedule_and_actual_acoustic_counts_are_bound(self):
        ids,rows=self.steps();self.assertTrue(validate_exposure(rows,ids,49*16))
        for change in ['omit','duplicate','user','weight','codebook','count']:
            bad=copy.deepcopy(rows)
            if change=='omit':bad[-1]['assistant_audio49_microbatch'].pop()
            elif change=='duplicate':bad[0]['assistant_audio49_microbatch'][0]['id']=ids[1]
            elif change=='user':bad[0]['assistant_audio49_microbatch'][0]['objective']['user_audio_loss_targets']=1
            elif change=='weight':bad[0]['assistant_audio49_microbatch'][0]['weight']=1.
            elif change=='codebook':bad[0]['assistant_audio49_microbatch'][0]['objective']['assistant_audio_targets'].pop()
            else:bad[0]['assistant_audio49_microbatch'][0]['objective']['assistant_audio_targets'][0]+=1
            with self.subTest(change=change),self.assertRaises(ValueError):validate_exposure(bad,ids,49*16)

    def test_smallstep_is_explicit_and_cannot_change_objectives_or_extend(self):
        a=self.args();a.native_audio49_step_scale=.1;self.assertTrue(validate_configuration(a))
        for field,value in [('actual_fit_updates',32),('actual_base_retention_weight',0.),('native_v2_learning_rate',2e-7),('native_audio49_step_scale',.01)]:
            b=self.args();b.native_audio49_step_scale=.1;setattr(b,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):validate_configuration(b)
    def test_smallstep_cannot_be_enabled_without_approved_audio49(self):
        a=SimpleNamespace(native_audio49_step_scale=.1)
        with self.assertRaises(ValueError):validate_configuration(a)
