import unittest
from data.synthetic.reasoning_audio1072_alignment import measured_words

class AlignmentTest(unittest.TestCase):
    def test_measured_words_preserved(self):
        words=[dict(text='Five',start=.1,end=.4),dict(text='pounds.',start=.5,end=.9)]
        result=measured_words('Five pounds.',dict(words=words),1.)
        self.assertTrue(result)
        self.assertEqual(words[1]['end'],.9)
    def test_numeric_source_token_keeps_single_real_interval(self):
        words=[dict(text='22',start=.1,end=.5),dict(text=' ',start=.5,end=.6),dict(text='washers.',start=.6,end=1.)]
        result=measured_words('22 washers.',dict(words=words),1.1)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0],words[0])

    def test_bad_measurement_or_missing_word_rejected(self):
        for words in [[dict(text='Five',start=.1,end=.4)], [dict(text='Five pounds.',start=.1,end=2.)]]:
            with self.assertRaises(ValueError):measured_words('Five pounds.',dict(words=words),1.)
