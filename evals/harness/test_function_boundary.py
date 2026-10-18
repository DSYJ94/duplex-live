import unittest
from training.control.function_boundary import name_phase
class FunctionOutputBoundary(unittest.TestCase):
 def test_only_incomplete_first_function_header_is_active(self):
  for text in ('<function=','\n<function=resolve','<function=resolve_pending_speech'):
   self.assertTrue(name_phase(text))
  for text in ('','<','<function','<function=answer>','<function=answer>\n<parameter=content>','<function=answer><parameter=content><function=resolve','<parameter=content><function=answer','<function=answer\n<parameter=content','<function=answer=content'):
   self.assertFalse(name_phase(text))
