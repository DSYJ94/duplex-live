import copy
import unittest
from data.synthetic.native_control_repair import build
from evals.native_control_training import schedule, exposure
from evals.native_control_cache import FAMILIES


class ControlTraining(unittest.TestCase):
    def records(self):
        return [dict(id=r['id'],family=r['family']) for r in build()['cases']]

    def steps(self, policy):
        return [dict(step=i+1,control40_microbatch=[dict(id=r['id'],family=r['family'],weight=r['weight_when_selected'],
            text_ce=1.,acoustic_ce=2.,text_targets=400,prefix_span_user_audio_loss_targets=0,
            acoustic_objective=dict(assistant_audio_targets=[400]+[399]*7,user_audio_loss_targets=0))
            for r in policy['teachers'][i::8]]) for i in range(8)]

    def test_actual_authored40_are_all_covered_with_every_family_each_update(self):
        records=self.records();declared=schedule(list(reversed(records)));policy=dict(teachers=declared)
        self.assertTrue(exposure(self.steps(policy),policy))
        self.assertEqual({r['id'] for r in declared},{r['id'] for r in records})
        for i in range(8):
            self.assertEqual({r['family'] for r in declared[i::8]},FAMILIES)
            self.assertAlmostEqual(sum(r['weight_when_selected'] for r in declared[i::8]),1.)
        for invalid in (records[:-1],records[:-1]+records[:1]):
            with self.assertRaises(ValueError):schedule(invalid)

    def test_exposure_rejects_subset_changed_weights_user_targets_and_missing_audio_heads(self):
        policy=dict(teachers=schedule(self.records()));original=self.steps(policy)
        for mode in ('subset','weight','user','heads','nan'):
            steps=copy.deepcopy(original);row=steps[0]['control40_microbatch'][0]
            if mode=='subset':steps[-1]['control40_microbatch'].pop()
            if mode=='weight':row['weight']=1.
            if mode=='user':row['acoustic_objective']['user_audio_loss_targets']=1
            if mode=='heads':row['acoustic_objective']['assistant_audio_targets'].pop()
            if mode=='nan':row['text_ce']=float('nan')
            with self.subTest(mode=mode),self.assertRaises(ValueError):exposure(steps,policy)


if __name__=='__main__':unittest.main()
