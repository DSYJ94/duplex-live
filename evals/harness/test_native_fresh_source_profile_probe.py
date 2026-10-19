import copy
import unittest
from evals.native_fresh_source_profile_probe import agreement,require_same_cases


class ProposedProfileProbeTests(unittest.TestCase):
    def test_only_compound_spelling_removes_lookup_segmentation_error(self):
        r=agreement('Cancel that lookup.','Cancel that look up.')
        self.assertGreater(r['raw_wer'],.05);self.assertEqual(r['proposed_spelling_wer'],0.)

    def test_word_substitutions_and_omission_stay_failed(self):
        for a,b in [('room sign','rune sign'),('warm day','worn day'),('Check the room sign.','Check room sign.'),('Go.','Go Go.')]:
            with self.subTest(a=a,b=b):self.assertGreater(agreement(a,b)['proposed_spelling_wer'],.05)

    def test_missing_duplicate_or_reordered_cases_cannot_form_whole20(self):
        m=dict(rows=[dict(scenario_id='native-fresh2-%03d'%i) for i in range(20)])
        d=dict(results=copy.deepcopy(m['rows']));require_same_cases(m,d)
        for rows in (d['results'][:-1],d['results'][:-1]+[d['results'][0]],list(reversed(d['results']))):
            with self.assertRaises(ValueError):require_same_cases(m,dict(results=rows))


if __name__=='__main__':unittest.main()
