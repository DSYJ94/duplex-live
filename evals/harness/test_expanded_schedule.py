import unittest
from collections import Counter
from training.personaplex.expanded_schedule import build
from training.personaplex.matched_schedule import SIZES


class ExpandedSchedule(unittest.TestCase):
    def fixture(self):
        groups={g:[dict(id=g+str(i),fold='TRAIN',kind='same') for i in range(n)]for g,n in SIZES.items()}
        rows=[dict(id='new'+str(i),fold='TRAIN' if i<1008 else 'DEV')for i in range(1072)]
        return groups,rows
    def test_complete_coverage_and_exact_global_mean_mass(self):
        groups,rows=self.fixture();p=build(groups,rows,world_size=4)
        self.assertEqual(p['total_new_updates'],196)
        for cycle in p['schedule']:
            seen=Counter();mass=Counter()
            for step in cycle:
                for slot in step['microbatches']:
                    self.assertEqual(len(slot),4)
                    for r in slot:
                        if r['valid']:seen[r['id']]+=1;mass[r['id']]+=step['new_coefficient']/step['new_global_valid_count']/49
                self.assertIn(step['new_global_valid_count'],(20,21))
            self.assertEqual(set(seen),{'new'+str(i)for i in range(1008)})
            self.assertTrue(all(v==1 for v in seen.values()))
            for value in mass.values():self.assertAlmostEqual(value,1/1008)
    def test_deterministic_resume_schedule_and_no_dev_insertion(self):
        groups,rows=self.fixture();a=build(groups,rows,world_size=4);self.assertEqual(a,build(groups,rows,world_size=4))
        self.assertNotEqual(a['schedule'][0][0]['new_global_ids'],a['schedule'][1][0]['new_global_ids'])
        rows[-1]['fold']='TRAIN'
        with self.assertRaises(ValueError):build(groups,rows,world_size=4)


if __name__=='__main__':unittest.main()
