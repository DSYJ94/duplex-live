import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.immutable_file_hashes import verified_reads,active
from evals.moshi_model_download import sha256,_sha256_uncached


class ImmutableInputHashes(unittest.TestCase):
    def test_full_first_hash_reused_without_caching_generated_outputs_or_default_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);inputs=root/'inputs';inputs.mkdir();fixed=inputs/'waveform';fixed.write_bytes(b'original')
            out=root/'state';out.write_bytes(b'first')
            with patch('evals.moshi_model_download._sha256_uncached',wraps=_sha256_uncached) as read:
                sha256(fixed);sha256(fixed);self.assertEqual(read.call_count,2)
                with verified_reads([inputs]) as session:
                    expected=sha256(fixed);self.assertEqual(sha256(fixed),expected)
                    self.assertEqual(read.call_count,3)
                    first=sha256(out);out.write_bytes(b'next');self.assertNotEqual(sha256(out),first)
                self.assertEqual(session.hits,1);self.assertTrue(session.receipt()['all_cached_identities_rechecked'])
            self.assertIsNone(active())

    def test_same_length_mutation_is_rejected_even_when_mtime_is_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'data';p.write_bytes(b'one');before=p.stat()
            with self.assertRaises(ValueError),verified_reads([root]):
                sha256(p);p.write_bytes(b'two');os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns));sha256(p)
            self.assertIsNone(active())

    def test_atomic_replacement_and_symlink_retarget_are_rejected(self):
        for mode in ('replace','retarget'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);p=root/'data';p.write_bytes(b'one');link=root/'alias';link.symlink_to(p)
                with self.assertRaises(ValueError),verified_reads([root]):
                    sha256(link);other=root/'other';other.write_bytes(b'one')
                    if mode=='replace':other.replace(p)
                    else:link.unlink();link.symlink_to(other)
                    sha256(link)

    def test_end_check_rejects_mutation_after_last_hash_and_exception_unwinds_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=root/'data';p.write_bytes(b'original')
            with self.assertRaises(ValueError),verified_reads([root]):
                sha256(p);p.write_bytes(b'changed')
            self.assertIsNone(active())
            with self.assertRaises(RuntimeError),verified_reads([root]):raise RuntimeError('stage failed')
            self.assertIsNone(active())


if __name__=='__main__':unittest.main()
