"""Counterexample from actual replacement failure: dispatch is not completion."""
import numpy as np
import unittest
from evals.harness import test_native_reasoning_stream as original
from evals import native_reasoning_stream_v2 as revised


class CompletionAwareStream(original.ReasoningStreamTests):
    capture_impl=staticmethod(revised.capture)

    async def test_slow_provider_and_delayed_context_ack_get_the_full_response_window(self):
        socket=original.Socket(provider_delay_s=8.,ack_delay_s=3.)
        report,events,ledger,*_=await self.run_capture(socket,[np.full(3840,123,dtype='<i2')])
        self.assertTrue(report['transport_complete'],report['errors'])
        finished=ledger['tasks']['original']['provider_finished_at_s']
        ack=next(e['received_at_s'] for e in events if e['type']=='session.commentary.appended')
        self.assertGreater(ack,finished)
        self.assertGreaterEqual(report['server_closed_at_s'],ack+revised.RESPONSE_WINDOW_S)
        self.assertFalse(report['training_approved'])


class OriginalCloseCounterexample(unittest.IsolatedAsyncioTestCase):
    run_capture=original.ReasoningStreamTests.run_capture
    capture_impl=staticmethod(original.capture)
    async def test_original_closed_before_late_ack_and_cannot_qualify(self):
        socket=original.Socket(provider_delay_s=8.,ack_delay_s=3.)
        report,events,ledger,*_=await self.run_capture(socket,[np.full(3840,123,dtype='<i2')])
        self.assertFalse(report['transport_complete'])
        self.assertLess(report['server_closed_at_s'],ledger['tasks']['original']['provider_finished_at_s']+revised.RESPONSE_WINDOW_S)


if __name__=='__main__':__import__('unittest').main()
