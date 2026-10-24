import unittest
from portable.context import native_control_copyback as backup


class ReasoningCopyback(unittest.TestCase):
    def test_fixed_reasoning_profile_excludes_other_roots_and_sensitive_files(self):
        try:
            backup.select_profile(True)
            for root in backup.REASONING_ROOTS:
                self.assertEqual(str(backup.safe_name(root+'/report.json')),root+'/report.json')
            for name in ('native-control40-precise-pilot-v2/training/report.json',
                         'native-reasoning32-retained-pilot-v15/.private/key',
                         'native-reasoning32-retained-pilot-v15/owner-conversations/audio.wav',
                         'native-reasoning32-retained-pilot-v15/session-token',
                         'native-reasoning32-retained-pilot-v15/core.123',
                         'native-reasoning32-retained-pilot-v15/training.tmp',
                         'native-reasoning32-retained-pilot-v15/../secret'):
                with self.subTest(name=name),self.assertRaises(ValueError):backup.safe_name(name)
            self.assertEqual(str(backup.EPHEMERAL_SNAPSHOTS),'/tmp/duplex-reasoning32-backup-snapshots')
        finally:
            backup.select_profile(False)
        self.assertEqual(backup.ROOTS,backup.CONTROL_ROOTS)


if __name__=='__main__':unittest.main()
