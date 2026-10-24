"""Real filesystem corruption/escape fixtures, with no model or GPU imports."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.native_gcp_destination_verify import _verify_entries, verify, main


def entry(path, content=b'checkpoint'):
    return dict(path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


class DestinationTests(unittest.TestCase):
    def test_all_files_and_archival_links_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'state.pt').write_bytes(b'checkpoint')
            (root/'python').symlink_to('/usr/bin/python')
            (root/'unrelated').write_text('not in the catalog')
            report = _verify_entries(root, [entry('state.pt')],
                                     [dict(path='python', target='/usr/bin/python')])
            self.assertTrue(report['passed'])
            self.assertEqual(report['verified_regular_files'], 1)
            self.assertEqual(report['verified_bytes'], 10)
            self.assertEqual(report['verified_archival_links'], 1)

    def test_same_size_corruption_and_missing_file_both_fail_no_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'a').write_bytes(b'checkpoinX')
            (root/'c').write_bytes(b'checkpoint')
            report = _verify_entries(root, [entry('a'), entry('b'), entry('c')], [])
            self.assertFalse(report['passed'])
            self.assertEqual(report['verified_regular_files'], 1)
            self.assertEqual([r['path'] for r in report['failures']], ['a', 'b'])
            self.assertEqual(report['failures'][0]['reason'], 'sha256_mismatch')

    def test_parent_and_final_file_symlinks_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'root'; root.mkdir()
            outside = Path(directory)/'outside'; outside.mkdir()
            (outside/'state.pt').write_bytes(b'checkpoint')
            (root/'parent').symlink_to(outside, target_is_directory=True)
            (root/'state.pt').symlink_to(outside/'state.pt')
            report = _verify_entries(root, [entry('parent/state.pt'), entry('state.pt')], [])
            self.assertFalse(report['passed'])
            self.assertEqual(report['verified_regular_files'], 0)
            self.assertEqual(len(report['failures']), 2)

    def test_archival_target_must_match_even_when_both_targets_are_python(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'python').symlink_to('/usr/bin/python3')
            report = _verify_entries(root, [], [dict(path='python', target='/usr/bin/python')])
            self.assertFalse(report['passed'])
            self.assertEqual(report['failures'][0]['reason'], 'archival_link_target_mismatch')

    def test_hardlinked_regular_snapshot_files_are_allowed_but_fifos_are_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'a').write_bytes(b'checkpoint')
            os.link(root/'a', root/'b')
            report = _verify_entries(root, [entry('a'), entry('b')], [])
            self.assertTrue(report['passed'])
            os.mkfifo(root/'fifo')
            report = _verify_entries(root, [entry('fifo')], [])
            self.assertFalse(report['passed'])
            self.assertEqual(report['failures'][0]['reason'], 'regular_file_required')

    def test_a_file_changed_while_hashing_cannot_receive_a_passing_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root/'state.pt'; state.write_bytes(b'checkpoint')
            real_digest = hashlib.sha256()

            class MutatingDigest:
                def update(self, chunk):
                    real_digest.update(chunk)
                    state.write_bytes(b'checkpoinX')

                def hexdigest(self):
                    return real_digest.hexdigest()

            expected = entry('state.pt')
            with patch('evals.native_gcp_destination_verify.hashlib.sha256', return_value=MutatingDigest()):
                report = _verify_entries(root, [expected], [])
            self.assertFalse(report['passed'])
            self.assertEqual(report['failures'][0]['reason'], 'file_changed_during_hash')

    def test_symlink_root_and_duplicate_file_link_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'root'; root.mkdir()
            alias = Path(directory)/'alias'; alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Physical destination'):
                _verify_entries(alias, [], [])
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                _verify_entries(root, [entry('state.pt')], [dict(path='state.pt', target='lib')])

    def test_production_entry_rejects_partial_or_changed_catalog_before_hashing(self):
        with tempfile.TemporaryDirectory() as directory:
            catalog = Path(directory)/'partial.json.gz'; catalog.write_bytes(b'{}')
            with patch('evals.native_gcp_destination_verify._verify_entries') as worker:
                with self.assertRaisesRegex(ValueError, 'inventory bytes changed'):
                    verify(directory, catalog)
                worker.assert_not_called()

    def test_existing_receipt_is_preserved_without_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)/'receipt.json'; report.write_text('original')
            with patch('sys.argv', ['verify', '--root', directory, '--inventory', 'unused',
                                    '--report', str(report)]), patch(
                    'evals.native_gcp_destination_verify.verify') as worker:
                with self.assertRaises(FileExistsError): main()
                worker.assert_not_called()
            self.assertEqual(report.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()
