"""Exercise the real host hash entry without requiring Python 3.7 contextvars."""
import builtins
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from evals.moshi_model_download import sha256


class LegacyHostHash(unittest.TestCase):
    def old_host(self):
        original = builtins.__import__
        def unavailable(name, *args, **kwargs):
            if name in ('contextvars', 'evals.immutable_file_hashes'):
                raise ModuleNotFoundError('Python3.6 has no contextvars')
            return original(name, *args, **kwargs)
        return patch('builtins.__import__', side_effect=unavailable)

    def test_legacy_host_reads_full_bytes_without_importing_modern_hash_session(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'proof.json'
            path.write_bytes(b'original native proof')
            with patch.object(sys, 'version_info', (3, 6, 15)), self.old_host():
                self.assertEqual(sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
                path.write_bytes(b'changed native proof')
                self.assertEqual(sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())

    def test_legacy_host_does_not_hide_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'version_info', (3, 6, 15)), self.old_host():
                with self.assertRaises(FileNotFoundError):
                    sha256(Path(directory) / 'missing')


if __name__ == '__main__':
    unittest.main()
