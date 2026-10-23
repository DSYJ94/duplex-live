import unittest
from evals import native_reasoning_teacher_ack_probe as probe
from evals import native_reasoning_stream_v6 as stream
from evals import native_reasoning_teacher_v7 as recorder
from evals import native_reasoning_teacher_cutoff_readback as readback


class AckProbeTests(unittest.TestCase):
    def test_only_acknowledgment_policy_changes(self):
        before=stream.configuration();after=probe.configuration()
        self.assertEqual({k:v for k,v in before.items()if k!='instructions'},
                         {k:v for k,v in after.items()if k!='instructions'})
        self.assertEqual(after['instructions'],before['instructions'].replace(probe.BACKCHANNEL,probe.ACK))
        self.assertEqual(after['instructions'].split('Delegation policy:')[1],
                         before['instructions'].split('Delegation policy:')[1])
        self.assertNotIn('eighteen',probe.ACK);self.assertNotIn('SpawnThinking',probe.ACK)

    def test_exact_transport_and_corrected_independent_readback_remain(self):
        capture,verify=probe.implementation()
        self.assertIs(capture.__code__,recorder.capture_one.__code__)
        self.assertIs(capture.__globals__['capture'].__code__,stream.capture.__code__)
        self.assertIs(verify.__code__,readback.verify_record.__code__)
        self.assertEqual(capture.__globals__['configuration'](),verify.__globals__['configuration']())
        self.assertEqual(stream.configuration()['instructions'].count(probe.BACKCHANNEL),1)


if __name__=='__main__':unittest.main()
