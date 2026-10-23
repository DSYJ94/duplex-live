import unittest
from evals import native_reasoning_teacher_v8 as revised
from evals import native_reasoning_teacher_v7 as original
from evals import native_reasoning_stream_v6 as stream
from evals import native_reasoning_teacher_cutoff_readback as readback


class RecorderTests(unittest.TestCase):
    def test_cutoff_only_profile_preserves_configuration_and_transport(self):
        capture,verify=revised.implementation(False)
        self.assertEqual(capture.__globals__['configuration'](),stream.configuration())
        self.assertIs(capture.__code__,original.capture_one.__code__)
        self.assertIs(capture.__globals__['capture'].__code__,stream.capture.__code__)
        self.assertIs(verify.__code__,readback.verify_record.__code__)

    def test_clarity_changes_only_instructions_and_never_global_functions(self):
        before=stream.configuration(); original_capture=original.capture_one
        capture,verify=revised.implementation(True)
        config=capture.__globals__['configuration']()
        self.assertEqual({k:v for k,v in config.items()if k!='instructions'},
                         {k:v for k,v in before.items()if k!='instructions'})
        self.assertTrue(config['instructions'].startswith(before['instructions']))
        self.assertEqual(config['instructions'],verify.__globals__['configuration']()['instructions'])
        self.assertIs(original.capture_one,original_capture)
        self.assertEqual(stream.configuration(),before)
        self.assertNotIn('48',revised.CLARITY)
        self.assertNotIn('SpawnThinking',revised.CLARITY)

    def test_corrected_readback_and_wrapper_are_bound_in_environment(self):
        capture,verify=revised.implementation(True)
        self.assertEqual(capture.__globals__['code_hashes'](),verify.__globals__['code_hashes']())
        self.assertIn('native_reasoning_teacher_cutoff_readback.py',revised.code_hashes())
        self.assertIn('native_reasoning_teacher_v8.py',revised.code_hashes())
        with self.assertRaises(ValueError):revised.implementation('true')


if __name__=='__main__':unittest.main()
