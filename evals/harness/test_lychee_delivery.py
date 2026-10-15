import unittest
from types import SimpleNamespace
from evals.lychee_context import audit_next_request
from evals.lychee_delivery import QuietDelivery


class DeliveryTests(unittest.TestCase):
    def test_quiet_user_is_not_enough_while_assistant_speaks(self):
        gate=QuietDelivery('idle-quiet-diagnostic')
        self.assertFalse(gate.observe(True,'s'))
        self.assertFalse(gate.observe(True,'s'))
        self.assertTrue(gate.observe(True,'l'))
        self.assertFalse(gate.observe(False,'l'))
        self.assertFalse(gate.observe(True,'l'))
        self.assertTrue(gate.observe(True,'l'))

    def test_actual_engine_rejects_stale_prefix(self):
        seen=[]
        original=lambda *a,**k:seen.append(a)
        engine=SimpleNamespace(_add_request_compat=original)
        proof={}
        audit_next_request(engine,[11,12],proof)
        with self.assertRaisesRegex(RuntimeError,'refreshed system prefix'):
            engine._add_request_compat('new',[11,99,13])
        self.assertEqual(seen,[])
        self.assertFalse(proof['actual_engine_request_verified'])
        self.assertIs(engine._add_request_compat,original)
        audit_next_request(engine,[11,12],proof)
        engine._add_request_compat('new',[11,12,13])
        self.assertTrue(proof['actual_engine_request_verified'])
        self.assertEqual(proof['actual_engine_prompt_tokens'],3)
        self.assertEqual(len(seen),1)


if __name__=='__main__':unittest.main()
