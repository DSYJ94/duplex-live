"""Independent enumeration and input-binding checks, never benchmark constants."""
import itertools
import unittest
from evals.exact_count_tool import execute, integer_words


def args(numbers, target):
    return dict(numbers=numbers, target=target, source_numbers=str(numbers), source_target=str(target))


class ExactCountTest(unittest.TestCase):
    def test_counts_match_exhaustive_enumeration_including_zero(self):
        for numbers in ([1, 2, 3], [2, 5, 9, 12], [3, 4, 7, 11, 18]):
            for target in range(sum(numbers) + 2):
                a = args(list(numbers), target)
                question = f'Numbers {a["source_numbers"]}; target {target}.'
                expected = sum(sum(c) == target for size in range(len(numbers)+1) for c in itertools.combinations(numbers, size))
                self.assertEqual(execute(question, a)['value'], expected)

    def test_changed_or_unquoted_values_are_rejected(self):
        a = args([2, 5, 9], 7)
        question = 'Numbers [2, 5, 9]; target 7.'
        for changed in (dict(a, numbers=[2, 5, 8]), dict(a, target=8), dict(a, source_numbers='[1, 2, 3]')):
            with self.assertRaises(ValueError):
                execute(question, changed)

    def test_resource_bounds_duplicates_boolean_and_negative_values_rejected(self):
        for numbers, target in [([1, 1], 2), ([True, 2], 2), ([-1, 2], 1), (list(range(1, 26)), 50), ([1, 2], 100001), ([1, 2], True)]:
            a = args(numbers, target)
            with self.assertRaises(ValueError):
                execute(f'{a["source_numbers"]}; {a["source_target"]}', a)

    def test_quotes_cannot_turn_negative_decimal_or_fraction_values_into_integers(self):
        for quote, values in [('[-3, 5, 8]', [3, 5, 8]), ('[1.5, 2]', [1, 5, 2]), ('[3/4, 2]', [3, 4, 2]), ('3-7', [3, 7])]:
            a = dict(numbers=values, target=2, source_numbers=quote, source_target='two')
            with self.assertRaises(ValueError):
                execute(f'Numbers {quote}; target two.', a)

    def test_spoken_number_list_with_final_and_is_bound_without_changing_values(self):
        a = dict(numbers=[2, 5, 9], target=7, source_numbers='two, five, and nine', source_target='seven')
        self.assertEqual(execute('For two, five, and nine, count subsets summing to seven.', a)['value'], 1)

    def test_partial_quotes_do_not_hide_signs_or_truncate_digits(self):
        a = dict(numbers=[3, 5, 8], target=2, source_numbers='3, 5, 8', source_target='2')
        with self.assertRaises(ValueError):
            execute('Numbers [-3, 5, 8], target 2.', a)
        a = dict(numbers=[2, 5, 9], target=3, source_numbers='[2, 5, 9]', source_target='3')
        with self.assertRaises(ValueError):
            execute('Numbers [2, 5, 9], target 31.', a)

    def test_spoken_integer_format_has_no_model_supplied_number(self):
        self.assertEqual(integer_words(0), 'zero')
        self.assertEqual(integer_words(101), 'one hundred one')
        self.assertEqual(integer_words(1000001), 'one million one')
        self.assertEqual(integer_words(2**24), 'sixteen million seven hundred seventy seven thousand two hundred sixteen')


if __name__ == '__main__':
    unittest.main()
