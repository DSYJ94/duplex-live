import unittest
from evals.context_retention_teacher import select_teachers


class TeacherInventoryTest(unittest.TestCase):
    def scene(self):
        return [{'record':dict(id=kind,kind=kind,source_model='gpt-live-1',source_split='train')}
            for kind in ('private','simple','clarify','cancel')]

    def test_exhaustive_independent_teachers_and_first_four_kinds(self):
        rows=self.scene()
        rows.extend([{'record':dict(id='private-2',kind='private',source_model='gpt-live-1',source_split='train')},
            {'record':dict(id='synthetic',kind='simple',source_model='synthetic',source_split='train')}])
        selected=select_teachers(rows,True)
        self.assertEqual([r['record']['id'] for r in selected],['private','simple','clarify','cancel','private-2'])
        self.assertEqual(len({r['record']['id'] for r in selected}),5)
        self.assertEqual(select_teachers(list(reversed(rows)),True),selected)

    def test_held_out_duplicate_and_missing_control_sources_rejected(self):
        for change in ('heldout','duplicate','missing','kind'):
            rows=self.scene()
            if change=='heldout':rows[0]['record']['source_split']='validation'
            if change=='duplicate':rows.append(rows[0])
            if change=='missing':rows.pop()
            if change=='kind':rows[0]['record']['kind']='unknown'
            with self.assertRaises(RuntimeError):select_teachers(rows,True)


if __name__=='__main__':unittest.main()
