import unittest
from evals.native_fresh_asr_control import agreement


class FreshASRControlTests(unittest.TestCase):
    def test_only_spelling_changes_not_warm_worn_are_equivalent(self):
        a=agreement('What colour is this kite?','What color is this kite?')
        self.assertGreater(a['raw_wer'],.05);self.assertEqual(a['spelling_only_wer'],0.)
        for text in ('Why a worn day?', 'Warm warm day.', 'A warm day.'):
            self.assertGreater(agreement('Why a warm day?',text)['spelling_only_wer'],.05)


if __name__=='__main__':unittest.main()
