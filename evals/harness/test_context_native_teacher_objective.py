from collections import Counter
import unittest
from evals.context_native_teacher_objective import teacher_index
from evals.context_component_kind import retention_index


class NativeTeacherObjectiveTest(unittest.TestCase):
    def test_four_teacher_control_is_exactly_the_existing_kind_clock(self):
        kinds=['a','b','c','d','e','f','g','h']
        teachers=['private','simple','clarify','cancel']
        for step in range(520):
            ri=retention_index(kinds,step,4,True)
            self.assertEqual(teacher_index(kinds,step,ri,teachers),ri)

    def test_all_teachers_are_seen_without_changing_task_kind_exposure(self):
        kinds=['a','b','c','d','e','f','g','h']
        teachers=['private','simple','clarify','cancel']+['private']*18+['simple']*3
        counts=Counter();bykind=Counter();trajectory=[]
        for step in range(520):
            ri=retention_index(kinds,step,4,True);i=teacher_index(kinds,step,ri,teachers)
            self.assertEqual(teachers[i],['private','simple','clarify','cancel'][ri])
            counts[i]+=1;bykind[teachers[i]]+=1;trajectory.append(i)
        self.assertEqual(len(counts),25);self.assertEqual(set(bykind.values()),{130})
        resumed=[teacher_index(kinds,step,retention_index(kinds,step,4,True),teachers) for step in range(8,520)]
        self.assertEqual(resumed,trajectory[8:])

    def test_wrong_kind_clock_and_missing_kind_are_rejected(self):
        with self.assertRaises(RuntimeError):teacher_index(['a','b'],0,1,['private','simple','clarify','cancel'])
        with self.assertRaises(RuntimeError):teacher_index(['a','b'],1,1,['private'])


if __name__=='__main__':unittest.main()
