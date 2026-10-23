import unittest
from evals import native_reasoning_teacher_quiet_probe as probe


class QuietProbeTests(unittest.TestCase):
    def test_only_backchannel_prompt_changes_and_actual_transport_stays_exact(self):
        before=probe.parent.stream.configuration();after=probe.configuration()
        self.assertEqual(after['instructions'],before['instructions'].replace(probe.parent.BACKCHANNEL,probe.POLICY))
        self.assertEqual(after['instructions'].split('Delegation policy:')[1],before['instructions'].split('Delegation policy:')[1])
        self.assertEqual({k:v for k,v in before.items()if k!='instructions'},{k:v for k,v in after.items()if k!='instructions'})
        capture,verify=probe.implementation()
        self.assertIs(capture.__code__,probe.parent.original.capture_one.__code__)
        self.assertIs(capture.__globals__['capture'].__code__,probe.parent.stream.capture.__code__)
        self.assertEqual(capture.__globals__['configuration'](),verify.__globals__['configuration']())
        self.assertNotIn('eighteen',probe.POLICY);self.assertNotIn('SpawnThinking',probe.POLICY)


if __name__=='__main__':unittest.main()
