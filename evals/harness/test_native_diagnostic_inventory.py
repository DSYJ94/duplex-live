import os
from pathlib import Path
import tempfile
import unittest

from portable.context.native_diagnostic_inventory import DRIVER_LINKS, inventory, verify


class DiagnosticInventoryTests(unittest.TestCase):
    def fixture(self, root):
        (root/'report.json').write_text('{"complete":true}\n')
        for name, target in DRIVER_LINKS.items():
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.symlink_to(target)

    def test_dangling_pinned_driver_links_are_metadata_and_not_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            expected = inventory(root)
            self.assertEqual(len(expected), 3)
            self.assertEqual(verify(root, expected)['symlinks'], 2)
            self.assertFalse(verify(root, expected)['model_qualified'])

    def test_changed_file_missing_file_and_extra_file_all_fail(self):
        for change in ('edit', 'missing', 'extra'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root)
                expected = inventory(root)
                if change == 'edit': (root/'report.json').write_text('changed')
                elif change == 'missing': (root/'report.json').unlink()
                else: (root/'extra').write_text('extra')
                with self.assertRaises(ValueError): verify(root, expected)

    def test_wrong_driver_target_and_arbitrary_external_link_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            path = root/next(iter(DRIVER_LINKS))
            path.unlink()
            path.symlink_to('/tmp/different-driver')
            with self.assertRaises(ValueError): inventory(root)
            path.unlink()
            (root/'external').symlink_to('/tmp/private-file')
            with self.assertRaises(ValueError): inventory(root)

    def test_extra_directory_link_is_rejected_without_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'external').symlink_to('/tmp', target_is_directory=True)
            with self.assertRaises(ValueError): inventory(root)

    def test_special_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.mkfifo(root/'pipe')
            with self.assertRaises(ValueError): inventory(root)
