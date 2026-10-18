import unittest
from evals.qwen35_deliberation_probe import after_thinking
class NativeDeliberationBoundary(unittest.TestCase):
 def test_only_tokens_after_actual_closure_are_actions(self):
  self.assertEqual(after_thinking([10,11,99,20,21],[99]),[20,21])
 def test_missing_duplicate_or_empty_final_fails_without_repair(self):
  for values,closing in [([10,11],[99]),([99,20,99,21],[99]),([10,99],[99]),([99,20],[99,100])]:
   with self.assertRaises(ValueError):after_thinking(values,closing)
