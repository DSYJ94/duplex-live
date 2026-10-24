from copy import deepcopy
import unittest
from evals.native_reasoning_cache import inventory,controls,FAMILIES
from evals.native_reasoning_timing_convert import SCHEMA
from evals.native_reasoning_teacher_mixed import IDS


class CompleteInventory(unittest.TestCase):
    def value(self):
        families=sorted(FAMILIES)
        return dict(schema=SCHEMA,complete32=True,training_approved=False,
            channel_order=['assistant','user'],accepted=[dict(scenario_id=k,dir=k,
                path=k+'/conversation.wav',family=families[i//8],split='train',
                scenario_class='native_reasoning4_teacher',native_frames=10,samples=19200)
                for i,k in enumerate(IDS)])

    def test_requires_complete32_and_all_four_families(self):
        v=self.value();self.assertEqual(len(inventory(v)),32)
        missing=deepcopy(v);missing['accepted'].pop()
        with self.assertRaises(ValueError):inventory(missing)
        wrong=deepcopy(v);wrong['accepted'][0]['family']=wrong['accepted'][-1]['family']
        with self.assertRaises(ValueError):inventory(wrong)

    def test_held_out_data_and_extra_pcm_frame_are_rejected(self):
        for key,value in (('split','validation'),('samples',19201)):
            v=self.value();v['accepted'][0][key]=value
            with self.assertRaises(ValueError):inventory(v)

    def test_cancel_preserves_two_actual_requests_and_one_real_delivery(self):
        side=dict(timeline=dict(events=[dict(kind='marker',text='',start_ms=80),
            dict(kind='marker',text='',start_ms=240)]),request_markers=[dict(marker_ms=80),dict(marker_ms=240)],
            reference_arrivals=[dict(ready_at_s=.4,text='Actual cancellation receipt')])
        self.assertEqual(controls(side,10,'cancel-reasoning')[0],[1,3])
        side['reference_arrivals']=[]
        with self.assertRaises(ValueError):controls(side,10,'cancel-reasoning')


if __name__=='__main__':unittest.main()
