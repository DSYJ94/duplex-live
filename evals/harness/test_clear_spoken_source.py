import unittest
from data.synthetic.clear_spoken_source import prepare

class ClearSourceTest(unittest.TestCase):
    def test_literal_operators_decimal_units_and_ordinals(self):
        self.assertEqual(prepare('15(.20) equals 3'),'fifteen times (zero point two zero) equals three')
        self.assertEqual(prepare('2000 x .25 equals 500'),'two thousand times zero point two five equals five hundred')
        self.assertEqual(prepare('36th; 3/4ths; 60 mph'),'thirty sixth; three quarters; sixty miles per hour')
        self.assertEqual(prepare('250 ml; 4 kg per kg'),'two hundred and fifty millilitres; four kilograms per kilogram')
    def test_no_unit_substitution_inside_time_or_ordinary_words(self):
        self.assertEqual(prepare('six a.m. at home'),'six a.m. at home')
        self.assertEqual(prepare('50 cm³ of helium'),'fifty cubic centimetres of helium')
