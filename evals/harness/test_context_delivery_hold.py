import unittest
from evals.context_delivery_hold import hold_until
from evals.recovery_worker import RecordWorld


class DeliveryHoldTest(unittest.TestCase):
    def result(self):
        return {'route':'delegate','receipts':[{'tool':'lookup_record','executed':True}]}

    def test_real_lookup_waits_only_until_explicit_transport_deadline(self):
        self.assertEqual(hold_until(self.result(),2.,25.,3.),27.)
        self.assertIsNone(hold_until(self.result(),2.,25.,27.))
        self.assertIsNone(hold_until(self.result(),2.,0.,3.))

    def test_ordinary_answers_errors_and_cancel_are_never_held(self):
        for r in ({'route':'answer','receipts':[]},{'route':'error','receipts':[]},
                {'route':'cancel','receipts':[{'tool':'cancel_lookup'}]},
                {'route':'delegate','receipts':[{'tool':'lookup_record','executed':False}]}):
            self.assertIsNone(hold_until(r,2.,25.,3.))

    def test_invalid_clocks_are_rejected(self):
        for delay in (-1.,31.,float('nan'),float('inf')):
            with self.assertRaises(ValueError):hold_until(self.result(),2.,delay,3.)

    def test_cancellation_invalidates_completed_but_undelivered_receipt(self):
        world=RecordWorld({'Elm North':{'marker':'blue square'}})
        rev=world.reserve('lookup');receipt=world.read('lookup','Elm North','marker',rev)
        world.mark_ready(receipt)
        self.assertEqual(hold_until({'route':'delegate','receipts':[receipt]},2.,25.,3.),27.)
        world.cancel_many(['lookup'])
        self.assertFalse(world.deliverable(receipt))


if __name__=='__main__':unittest.main()
