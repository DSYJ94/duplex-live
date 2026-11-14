import unittest
from data.synthetic.math_spoken_words import words,wer


class MathSpeechOrthography(unittest.TestCase):
    def test_numbers_units_and_operators_have_fixed_spoken_forms(self):
        self.assertEqual(words('£573.44'),words('five hundred seventy three pounds and forty four pence'))
        self.assertEqual(words('$12.50'),words('twelve dollars and fifty cents'))
        self.assertEqual(words('2.4 metres'),words('two point four meters'))
        self.assertEqual(words('4*40=160'),words('four times forty equals one hundred sixty'))
    def test_sentence_currency_and_attached_unit_abbreviations(self):
        self.assertEqual(words('The answer is 15 pounds.'),words('The answer is £15.'))
        self.assertEqual(words('285 centimetres'),words('285cm'))
        self.assertEqual(words('88 metres'),words('88m'))
        self.assertNotEqual(words('285cm'),words('285mm'))

    def test_wrong_values_units_homophones_and_missing_words_fail(self):
        for a,b in [('Go.','Go. Go.'),('room sign','rune sign'),('12 pounds','12 dollars'),('$1.25','$125'),('4*40=160','4 40 160'),('4+4','8'),('A timer measures time','Timer measures time'),('The bus arrived first','The boss arrived first')]:
            with self.subTest(a=a,b=b):self.assertGreater(wer(a,b),.05)


if __name__=='__main__':unittest.main()
