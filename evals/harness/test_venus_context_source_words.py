import unittest
from data.venus.context_source_quality import words,wer


class SourceWordTests(unittest.TestCase):
    def test_money_and_numbers_have_fixed_equivalence(self):
        self.assertEqual(words('£17,000 at 8 percent over 36 months'),words('seventeen thousand pounds at eight percent over thirty-six months'))
        self.assertEqual(words('2.4 metres'),words('two point four meters'))
        self.assertEqual(wer('eighteen thousand three hundred pounds','£18,300'),0)

    def test_semantics_not_normalized_away(self):
        for a,b in [('room sign','rune sign'),('marker in cupboard','marker and cupboard'),('cancel travel','cancel meal'),('17000 pounds','18000 pounds'),('cobalt sunrise','amber river')]:
            self.assertGreater(wer(a,b),0)
