import copy
import unittest
from evals.context_native_audio49_data import complete_inventory,GROUPS

class Complete49AudioOwnership(unittest.TestCase):
    def rows(self):
        return [dict(group=g['name'],teacher=dict(id=g['name']+'-'+str(i),source_split='train',source_model='gpt-live-1'),
            assistant_audio_targets_per_head=8,user_audio_loss_targets=0,prefix_loss_targets=0,supplied_span_loss_targets=0)
            for g in GROUPS for i in range(g['count'])]
    def test_all25_and24_required(self):
        self.assertTrue(complete_inventory(self.rows()))
        with self.assertRaises(ValueError):complete_inventory(self.rows()[:-1])
    def test_duplicate_or_substituted_group_is_not_complete(self):
        for field,value in [('group','old25'),('teacher',self.rows()[0]['teacher'])]:
            r=self.rows();r[-1][field]=value
            with self.assertRaises(ValueError):complete_inventory(r)
    def test_no_user_prefix_or_span_targets(self):
        for field in ['user_audio_loss_targets','prefix_loss_targets','supplied_span_loss_targets']:
            r=self.rows();r[0][field]=1
            with self.assertRaises(ValueError):complete_inventory(r)
    def test_heldout_owner_or_empty_audio_targets_rejected(self):
        for change in ['heldout','owner','empty']:
            r=copy.deepcopy(self.rows())
            if change=='heldout':r[0]['teacher']['source_split']='validation'
            elif change=='owner':r[0]['teacher']['source_model']='owner'
            else:r[0]['assistant_audio_targets_per_head']=0
            with self.assertRaises(ValueError):complete_inventory(r)
