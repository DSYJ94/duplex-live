import unittest
from evals.context_component_kind import balanced_index,retention_index


class RetentionBalanceTest(unittest.TestCase):
    def test_each_of_six_kinds_sees_all_four_retention_examples(self):
        kinds=['cancel','clarify','lookup','ordinary','private','listen']
        seen={kind:[] for kind in kinds}
        for step in range(24):
            kind=kinds[balanced_index(kinds,step)]
            seen[kind].append(retention_index(kinds,step,4,balanced=True))
        self.assertTrue(all(sorted(v)==[0,1,2,3] for v in seen.values()),seen)

    def test_legacy_schedule_and_five_kind_pairing_remain_exact(self):
        for size in [5,6,7]:
            kinds=[str(i) for i in range(size)]
            for step in range(32):self.assertEqual(retention_index(kinds,step,4),step%4)
        for step in range(40):self.assertEqual(retention_index(list('abcde'),step,4,True),step%4)

    def test_invalid_scope_rejected(self):
        for kinds,step,count in [([],0,4),(['a'],-1,4),(['a'],0,0)]:
            with self.assertRaises(ValueError):retention_index(kinds,step,count,True)


if __name__=='__main__':unittest.main()
