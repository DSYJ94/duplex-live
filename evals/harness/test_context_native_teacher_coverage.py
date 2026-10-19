import unittest
from evals.context_native_teacher_coverage import complete_indices
from evals.context_component_kind import retention_index


class CompleteTeacherCoverageTests(unittest.TestCase):
    def test_every_teacher_and_quiet_pause_exposed_in_both_eight_update_blocks(self):
        kinds=['k'+str(i%8) for i in range(28)]
        teachers=['private']*8+['simple']*8+['clarify']*4+['cancel']*4
        for block in (0,8,16,24):
            seen=[];weights={k:0. for k in set(teachers)}
            for step in range(block,block+8):
                ri=retention_index(kinds,step,4,True)
                selected=complete_indices(kinds,step,ri,teachers)
                self.assertEqual(len(selected),4 if ri<2 else 2)
                seen.extend(selected)
                for index in selected:weights[teachers[index]]+=1/len(selected)
            self.assertEqual(sorted(seen),list(range(24)))
            self.assertEqual(set(weights.values()),{2.})

    def test_missing_kind_or_changed_counts_or_wrong_kind_clock_rejected(self):
        kinds=['k'+str(i) for i in range(8)]
        teachers=['private']*8+['simple']*8+['clarify']*4+['cancel']*4
        for changed in (teachers[:-1],teachers[:-1]+['private']):
            with self.assertRaises(ValueError):complete_indices(kinds,0,0,changed)
        with self.assertRaises(RuntimeError):complete_indices(kinds,0,1,teachers)


if __name__=='__main__':unittest.main()
