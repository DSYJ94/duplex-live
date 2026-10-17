import json
import tempfile
import unittest
from pathlib import Path
from evals.context_native_private import plan


class NativePrivateTest(unittest.TestCase):
    def test_same_sentence_same_prefix_but_no_future_receipts(self):
        import torch
        with tempfile.TemporaryDirectory() as name:
            h=Path(name)
            (h/'student-history.json').write_text(json.dumps({'source':{'split':'train','kind':'private'}}))
            rows=[{'history_virtual_frames_before_step':i,'history_virtual_frames_after_step':i+1} for i in range(12)]
            (h/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            state={'cache':torch.full((1,17,16),3),'provided':torch.zeros(1,17,16,dtype=torch.bool),
                'offset':3,'pending_exit_cb0':None}
            torch.save({'states':[state]*12},h/'student-history.pt')
            teacher={'id':'approved'};clean=torch.full((17,12),3);clean[0,4:8]=torch.tensor([0,42,43,3])
            late={'target_kind':'private_result','conditioning_policy':'verified-trained-rollout',
                'actual_private_receipt':{'executed':True},'actual_history':str(h),'teacher':teacher,
                'selection':{'source_start_frame':4,'source_end_frame':8},
                'future_first_real_frame':3,'future_frames':4,'replay_cut_index':0}
            m,gold=plan(late,[{'record':teacher,'clean_codes':clean}])
            self.assertTrue(torch.equal(gold,clean[:9,4:8]))
            self.assertEqual(m['future_first_real_frame'],3);self.assertEqual(m['future_end_real_frame'],7)
            self.assertEqual(m['future_recorded_receipts'],[])
            self.assertEqual(m['actual_prefix_last_predicted_text_token'],3)
            self.assertEqual(m['past_loss_targets']+m['audio_loss_targets']+m['user_loss_targets'],0)
            rows[4]['history_virtual_frames_before_step']+=20
            (h/'events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(RuntimeError):plan(late,[{'record':teacher,'clean_codes':clean}])
            (h/'student-history.json').write_text(json.dumps({'source':{'split':'validation','kind':'private'}}))
            with self.assertRaises(RuntimeError):plan(late,[{'record':teacher,'clean_codes':clean}])

    def test_scope_requires_verified_private_receipt(self):
        for value in ({},{'target_kind':'private_result','conditioning_policy':'verified-trained-rollout'}):
            with self.assertRaises(RuntimeError):plan(value,[])


if __name__=='__main__':unittest.main()
