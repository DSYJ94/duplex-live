import copy
import unittest
from evals.native_conditioning_probe import TEACHER,OLD_APPROVAL,policy
from evals.context_native_learning_curve import PINS

class ConditioningProbeTests(unittest.TestCase):
    def setUp(self):
        self.scope=dict(schema='dl-context-all-teacher-scope-audit-v2',passed=True,training_approved=False,optimizer_steps=0,
            teachers=[dict(id=TEACHER if i==0 else 'teacher-'+str(i),source_split='train') for i in range(25)])
        self.parent=dict(optimizer_steps=520,checkpoint_sha256='parent',voice_sha256='voice',assets_manifest_sha256='assets')
        self.child=dict(optimizer_steps=40,checkpoint_sha256=PINS[1]['checkpoint'],training_state_sha256=PINS[1]['state'],
            warm_start=dict(checkpoint_sha256='parent'),voice_sha256='voice',assets_manifest_sha256='assets')
    def test_diagnostic_is_never_training_or_behavior_approval(self):
        r=policy(self.scope,self.parent,self.child)
        self.assertTrue(r['teacher_text_forced']);self.assertFalse(r['agent_acoustics_forced'])
        self.assertTrue(r['outputs_must_not_be_trained_on']);self.assertFalse(r['training_approved'])
        self.assertEqual(r['optimizer_updates'],0);self.assertFalse(r['native_behavior_qualified'])
    def test_incomplete_duplicate_or_nontrain_scope_fails(self):
        for mode in ('incomplete','duplicate','heldout'):
            s=copy.deepcopy(self.scope)
            if mode=='incomplete':s['teachers'].pop()
            elif mode=='duplicate':s['teachers'][1]=s['teachers'][0]
            else:s['teachers'][0]['source_split']='validation'
            with self.assertRaises(ValueError):policy(s,self.parent,self.child)
    def test_different_or_extended_checkpoint_fails(self):
        for key,value in [('optimizer_steps',41),('checkpoint_sha256','other'),('training_state_sha256','other')]:
            c=copy.deepcopy(self.child);c[key]=value
            with self.assertRaises(ValueError):policy(self.scope,self.parent,c)
    def test_approval_path_matches_the_completed_old25_chain(self):
        self.assertEqual(OLD_APPROVAL,'native-old25-native-approval-7544805')

    def test_changed_voice_or_parent_fails(self):
        for key,value in [('voice_sha256','other'),('warm_start',dict(checkpoint_sha256='other'))]:
            c=copy.deepcopy(self.child);c[key]=value
            with self.assertRaises(ValueError):policy(self.scope,self.parent,c)
