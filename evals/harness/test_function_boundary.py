import unittest
from training.control.function_boundary import name_phase,target_positions
class FunctionOutputBoundary(unittest.TestCase):
 def test_only_incomplete_first_function_header_is_active(self):
  for text in ('<function=','\n<function=resolve','<function=resolve_pending_speech'):
   self.assertTrue(name_phase(text))
  for text in ('','<','<function','<function=answer>','<function=answer>\n<parameter=content>','<function=answer><parameter=content><function=resolve','<parameter=content><function=answer','<function=answer\n<parameter=content','<function=answer=content'):
   self.assertFalse(name_phase(text))

 def test_training_targets_match_cached_decode_and_exclude_argument_literals(self):
  # Pieces deliberately cross the closing delimiter. The correction predicting
  # that piece is active; the next argument token must use the original head.
  pieces=['<function=', 'resolve_', 'pending_speech>', '\n<parameter=content>',
          '<function=answer', '</parameter></function></tool_call>', '<eos>']
  ids=list(range(len(pieces)))
  self.assertEqual(target_positions(ids,lambda xs:''.join(pieces[x] for x in xs)),[1,2])
  with self.assertRaises(ValueError):
   target_positions([0,1],lambda xs:'<function=answer>' if xs else '')
