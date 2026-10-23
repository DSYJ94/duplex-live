"""Integrated delayed-control race, with native events and original PCM preserved."""
import unittest
import numpy as np
from evals.harness import test_native_reasoning_stream as original
from evals import native_reasoning_stream_v2 as previous
from evals import native_reasoning_stream_v3 as revised
from evals.harness import test_native_reasoning_teacher as artifacts
from evals import native_reasoning_teacher_v3 as recorder


class CausalStream(original.ReasoningStreamTests):
    capture_impl=staticmethod(revised.capture)

    async def test_slow_cancel_decision_holds_ready_original_without_forcing_request(self):
        socket=original.Socket(control=True,pending_original=False,provider_delays={'cancel':8.})
        report,events,ledger,calls,*_=await self.run_capture(socket,
            [np.full(3840,123,dtype='<i2'),np.full(3840,456,dtype='<i2')])
        self.assertTrue(report['transport_complete'],report['errors']);self.assertEqual(len(calls),2)
        self.assertTrue(any(e['type']=='delivery_barrier_wait' for e in ledger['events']))
        self.assertTrue(ledger['tasks']['original']['cancelled'])
        self.assertNotIn('wire_write_completed_at_s',ledger['tasks']['original'])
        self.assertEqual([e['delegation_id'] for e in events if e['type']=='local.commentary.write'],['cancel'])
        self.assertFalse(report['native_request_forced'])


class PreviousRace(unittest.IsolatedAsyncioTestCase):
    run_capture=original.ReasoningStreamTests.run_capture
    capture_impl=staticmethod(previous.capture)

    async def test_previous_recorder_reproduces_actual_stale_delivery_and_rejected_late_cancel(self):
        socket=original.Socket(control=True,pending_original=False,provider_delays={'cancel':8.})
        report,events,ledger,*_=await self.run_capture(socket,
            [np.full(3840,123,dtype='<i2'),np.full(3840,456,dtype='<i2')])
        self.assertFalse(report['transport_complete'])
        self.assertEqual(ledger['tasks']['original']['phase'],'wire_sent')
        self.assertEqual(ledger['tasks']['cancel']['phase'],'failed_preserved')


class CausalArtifacts(artifacts.TeacherArtifacts):
    implementation=recorder
    capture_stream=staticmethod(revised.capture)


if __name__=='__main__':unittest.main()
