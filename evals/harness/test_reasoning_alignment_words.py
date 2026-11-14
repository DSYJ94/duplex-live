import unittest
from data.synthetic.reasoning_audio1072_alignment import measured_units

class AlignmentTest(unittest.TestCase):
    def test_measured_words_preserved(self):
        words=[dict(text='Five',start=.1,end=.4),dict(text='pounds.',start=.5,end=.9)]
        result=measured_units('Five pounds.',dict(words=words),1.)
        self.assertTrue(result)
        self.assertEqual(words[1]['end'],.9)
    def test_bad_measurement_or_missing_word_rejected(self):
        for words in [[dict(text='Five',start=.1,end=.4)], [dict(text='Five pounds.',start=.1,end=2.)]]:
            with self.assertRaises(ValueError):measured_units('Five pounds.',dict(words=words),1.)
