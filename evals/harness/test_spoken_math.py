import unittest
from data.synthetic.spoken_math import verbalize


class SpokenMath(unittest.TestCase):
    def test_operators_and_units_are_explicit(self):
        self.assertEqual(verbalize('4*40=160 pizzas.'),'four times forty equals one hundred and sixty pizzas.')
        self.assertEqual(verbalize('4 marbles x 4 = 16marbles'),'four marbles times four equals sixteen marbles')
        self.assertEqual(verbalize('10 - 4 = 6'),'ten minus four equals six')
        self.assertEqual(verbalize('100-20=80'),'one hundred minus twenty equals eighty')
        self.assertEqual(verbalize('4x4=16'),'four times four equals sixteen')
    def test_money_and_decimals_preserve_values(self):
        self.assertEqual(verbalize('$12.50 and £1.01'),'twelve dollars and fifty cents and one pound and one penny')
        self.assertEqual(verbalize('2.4 metres'),'two point four metres')
    def test_no_semantic_repair_is_claimed(self):
        self.assertEqual(verbalize('13 + 4 + 16 = 45'),'thirteen plus four plus sixteen equals forty five')
        self.assertIn('bee-eaters',verbalize('10 bee-eaters'))
        self.assertEqual(verbalize('x is unknown'),'x is unknown')


if __name__=='__main__':unittest.main()
