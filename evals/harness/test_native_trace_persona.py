"""Exercise actual trace persona selection without loading model weights."""
from types import SimpleNamespace
import unittest
from evals.native_control_engine import trace
from evals.lychee_context import BASE_INSTRUCTIONS

class TracePersona(unittest.TestCase):
    def check_persona(self,expected,**kwargs):
        seen=[]
        graph=SimpleNamespace(_graph=None)
        gen=SimpleNamespace(_streaming_state=SimpleNamespace(graphed_main=graph),step=lambda:None,prepare_step_input=lambda:None)
        lm=SimpleNamespace(forward_codes=lambda:None)
        def persona(instructions,voice):
            seen.append((instructions,voice));raise RuntimeError('stop-after-real-persona-selection')
        engine=SimpleNamespace(lm=lm,lm_gen=gen,reset=lambda:None,set_persona=persona)
        with self.assertRaisesRegex(RuntimeError,'stop-after-real-persona-selection'):
            trace(engine,'voice',None,None,**kwargs)
        self.assertEqual(seen,[(expected,'voice')])
        self.assertIs(gen._streaming_state.graphed_main,graph)
    def test_explicit_persona_matches_new_capture_prefix(self):
        self.check_persona('New declared instructions',instructions='New declared instructions')
    def test_legacy_default_is_preserved(self):
        self.check_persona(BASE_INSTRUCTIONS)

if __name__=='__main__':unittest.main()
