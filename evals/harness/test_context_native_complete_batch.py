import copy
import math
import unittest
from evals.context_native_complete_batch import balanced_weights,validate_groups,validate_exposure,matches

class CompleteBalancedBatch(unittest.TestCase):
    def groups(self):
        kinds={'actual':['k'+str(i%8) for i in range(28)],'old25':['private']*8+['simple']*8+['clarify']*5+['cancel']*4,'new24':['private']*8+['simple']*8+['clarify']*4+['cancel']*4}
        groups={}
        for group,size in [('actual',28),('reference',4),('old25',25),('new24',24),('word',10),('clock',14),('audio49',49)]:
            ws=balanced_weights(kinds[group]) if group in kinds else [(64 if group in ('word','clock') else 1)/size]*size
            groups[group]=[dict(id=group+str(i),weight=w,**({'kind':kinds[group][i]} if group in kinds else {})) for i,w in enumerate(ws)]
        for row in groups['audio49']:row['objective']=dict(assistant_audio_targets=[2]*8,user_audio_loss_targets=0)
        return groups
    def test_every_kind_keeps_equal_mass_even_with_unequal_component_counts(self):
        kinds=['a']*3+['b'];weights=balanced_weights(kinds)
        self.assertAlmostEqual(sum(weights[:3]),.5);self.assertEqual(weights[-1],.5)
        self.assertTrue(validate_groups(self.groups()))
    def test_no_subset_duplicate_nonfinite_or_redistributed_kind_mass(self):
        for change in ('omit','duplicate','nan','redistribute'):
            g=self.groups()
            if change=='omit':g['old25'].pop()
            if change=='duplicate':g['new24'][1]['id']=g['new24'][0]['id']
            if change=='nan':g['audio49'][0]['weight']=math.nan
            if change=='redistribute':g['actual'][0]['weight']+=.01;g['actual'][-1]['weight']-=.01
            with self.subTest(change=change),self.assertRaises(ValueError):validate_groups(g)
    def test_eight_complete_audio_epochs_have_same_weight_mass_as_original_stride(self):
        g=self.groups();ids=[x['id'] for x in g['audio49']];steps=[dict(step=i+1,complete_balanced_microbatches=copy.deepcopy(g),assistant_audio49_microbatch=copy.deepcopy(g['audio49']),additional_native_v2_microbatch=copy.deepcopy(g['new24'])) for i in range(8)]
        self.assertTrue(validate_exposure(steps,ids,49*16,g))
        self.assertAlmostEqual(sum(s['assistant_audio49_microbatch'][0]['weight'] for s in steps),8/49)
        steps[-1]['complete_balanced_microbatches']['audio49'][0]['objective']['user_audio_loss_targets']=1
        with self.assertRaises(ValueError):validate_exposure(steps,ids,49*16,g)
    def test_a_changed_whole_membership_cannot_self_approve(self):
        g=self.groups();bad=copy.deepcopy(g);bad['old25'][0]['id']='unapproved'
        with self.assertRaises(ValueError):matches(bad,g)
