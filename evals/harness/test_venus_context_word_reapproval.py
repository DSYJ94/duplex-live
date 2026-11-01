import unittest
from data.venus.context_word_reapproval import wer

class OrthographyTests(unittest.TestCase):
 def test_only_specific_orthography_is_equivalent(self):
  self.assertEqual(wer('explain its main tradeoff briefly','explain its main trade-off briefly'),0)
  for a,b in [('re-sign','resign'),('room sign','rune sign'),('cancel travel','cancel meal'),('17000 pounds','18000 pounds')]:self.assertGreater(wer(a,b),0)
