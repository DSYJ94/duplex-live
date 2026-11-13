import copy
import unittest
from training.personaplex.distributed_stream_plan import align


class StreamPlan(unittest.TestCase):
    def test_unequal_streams_have_same_collectives_without_new_targets(self):
        rows=[[dict(start=0,frames=3,active_targets=0,phase='prefix'),dict(start=3,frames=2,active_targets=2)],
              [dict(start=0,frames=2,active_targets=0,phase='prefix'),dict(start=2,frames=1,active_targets=0),dict(start=3,frames=4,active_targets=4)]]
        before=copy.deepcopy(rows);p=align(rows)
        self.assertEqual(rows,before);self.assertEqual(len(p['slots']),3)
        self.assertEqual([s['backward_on_all_ranks'] for s in p['slots']],[False,True,True])
        self.assertTrue(p['slots'][2]['ranks'][0]['padding_only'])
        self.assertEqual(p['slots'][2]['ranks'][0]['active_targets'],0)
        for rank,original in enumerate(rows):
            self.assertEqual([s['ranks'][rank]['original_call'] for s in p['slots'] if not s['ranks'][rank]['padding_only']],original)
    def test_invalid_boundaries_and_prefix_target_rejected(self):
        for call in [dict(start=1,frames=2,active_targets=0),dict(start=0,frames=2,active_targets=1,phase='prefix')]:
            with self.assertRaises(ValueError):align([[call]])


if __name__=='__main__':unittest.main()
