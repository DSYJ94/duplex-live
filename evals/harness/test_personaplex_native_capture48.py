import unittest
from training.personaplex.native_capture48 import reference_sources


class ReferenceSelection(unittest.TestCase):
    def row(self,i,**values):
        return dict(id=str(i),profile='assistant-british',source_sha256='hash'+str(i),
                    fold='TRAIN',peak_pass=True,seconds=3.,**values)

    def test_complete_disjoint_same_profile_clips_in_stable_order(self):
        rows=[self.row(i) for i in (3,1,0,2)]
        rows.append(dict(rows[0],id='user',profile='user-british'))
        selected=reference_sources(rows,{'hash0'})
        self.assertEqual([r['id'] for r in selected],['1','2','3'])
        self.assertTrue(all(r['source_sha256']!='hash0' for r in selected))
        self.assertEqual(reference_sources(list(reversed(rows)),{'hash0'}),selected)

    def test_never_borrow_heldout_or_failed_sources(self):
        for change in ({'fold':'HOLDOUT'},{'peak_pass':False},{'seconds':float('nan')}):
            rows=[self.row(i) for i in range(4)];rows[0].update(change)
            with self.assertRaises(ValueError):reference_sources(rows,set())

    def test_no_fallback_to_short_or_wrong_speaker_reference(self):
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            reference_sources([self.row(0)],set())
        rows=[dict(self.row(i),profile='user-scottish') for i in range(5)]
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            reference_sources(rows,set())


if __name__=='__main__':unittest.main()
