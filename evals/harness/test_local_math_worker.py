import unittest
from portable.context.local_math_worker import calculate,run

class LocalArithmetic(unittest.TestCase):
    def test_rules_compute_from_question_not_expected_answers(self):
        cases=[('A workshop has 123 washers. It uses 5 and then receives 2. How many washers does it have now?','120'),('A cable measures 103 metres. After cutting off 35 centimetres, how many centimetres are left?','10265'),('A rectangle has length 107 metres and width 7 metres. What is its perimeter?','228'),('A resistor carries 2 amperes under 208 volts. Using voltage equals current times resistance, what is its resistance?','104')]
        for question,expected in cases:self.assertEqual(calculate(question)[0],expected)
        self.assertEqual(calculate(cases[0][0].replace('123','151'))[0],'148')
    def test_model_guesses_never_authorize_calculation(self):
        value=run([dict(role='assistant',text='A workshop has 123 washers. It uses 5 and then receives 2.')],None,'fixture')
        self.assertEqual(value['route'],'error');self.assertFalse(value['receipts'])
    def test_unsupported_question_is_honest_failure(self):
        value=run([dict(role='user',text='What is the current price of this stock?')],None,'fixture')
        self.assertEqual(value['route'],'error');self.assertIn('do not invent',value['content'])
