import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from training.personaplex import local_process_identity as p


class ProcessIdentity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / '123'
        self.proc.mkdir()
        boot = self.root / 'sys/kernel/random'
        boot.mkdir(parents=True)
        (boot / 'boot_id').write_text('fixture-boot\n')
        self.stat()
        (self.proc / 'status').write_text('Name:\tfixture\nUid:\t1001\t1001\t1001\t1001\n')
        (self.proc / 'cgroup').write_text('0::/slurm/job_1234/step_batch\n')

    def stat(self, start=987, state='S'):
        fields = [state] + ['0'] * 18 + [str(start)]
        (self.proc / 'stat').write_text('123 (name with ) parentheses) ' + ' '.join(fields))

    def test_same_process_and_complex_comm(self):
        expected = p.capture(123, self.root)
        self.assertEqual(expected['start_ticks'], 987)
        self.assertEqual(p.verify(expected, self.root), expected)

    def test_reused_pid_is_rejected(self):
        expected = p.capture(123, self.root)
        self.stat(start=988)
        with self.assertRaises(RuntimeError): p.verify(expected, self.root)

    def test_dead_or_missing_process_is_rejected(self):
        for state in ('Z', 'X', 'x'):
            self.stat(state=state)
            with self.assertRaises(RuntimeError): p.capture(123, self.root)
        (self.proc / 'stat').unlink()
        with self.assertRaises(OSError): p.capture(123, self.root)

    def test_reboot_uid_or_cgroup_change_is_rejected(self):
        expected = p.capture(123, self.root)
        for path in (self.root/'sys/kernel/random/boot_id', self.proc/'status', self.proc/'cgroup'):
            original = path.read_text()
            path.write_text(original.replace('1001', '1002') if path.name == 'status' else original + 'changed')
            with self.assertRaises(RuntimeError): p.verify(expected, self.root)
            path.write_text(original)

    def test_racing_pid_change_is_rejected(self):
        with patch.object(p, '_stat', side_effect=[987, 988]):
            with self.assertRaises(RuntimeError): p.capture(123, self.root)

    def test_incomplete_identity_is_rejected(self):
        with self.assertRaises(ValueError): p.verify({'pid': 123}, self.root)
        for value in (True, 0, -1, '123'):
            with self.assertRaises(ValueError): p.capture(value, self.root)
        (self.proc / 'status').write_text('Uid:\t1001\n')
        with self.assertRaises(RuntimeError): p.capture(123, self.root)


if __name__ == '__main__': unittest.main()
