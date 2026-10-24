import hashlib
from pathlib import Path
import tempfile
import unittest

from portable.context.native_gcp_inputs import regular_link, regular_copy_tree


class InputMaterializationTests(unittest.TestCase):
    def test_snapshot_hardlink_preserves_exact_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.write_bytes(b'checkpoint')
            expected=dict(bytes=10,sha256=hashlib.sha256(b'checkpoint').hexdigest())
            target=root/'snapshot/state.pt';regular_link(source,target,expected)
            self.assertEqual(source.stat().st_ino,target.stat().st_ino)
            with self.assertRaises(FileExistsError):regular_link(source,target,expected)

    def test_changed_source_or_symlink_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.write_bytes(b'checkpoint')
            with self.assertRaisesRegex(ValueError,'changed'):
                regular_link(source,root/'target',dict(bytes=10,sha256='0'*64))
            (root/'alias').symlink_to(root,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlinks'):
                regular_link(source,root/'alias/target')

    def test_temporary_inputs_are_regular_independent_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            (source/'teacher.json').write_text('{}')
            target=root/'durable/source'
            self.assertEqual(regular_copy_tree(source,target),1)
            self.assertEqual((target/'teacher.json').read_bytes(),(source/'teacher.json').read_bytes())
            self.assertNotEqual((target/'teacher.json').stat().st_ino,(source/'teacher.json').stat().st_ino)
            with self.assertRaises(FileExistsError):regular_copy_tree(source,target)

    def test_temporary_symlink_roots_and_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            (root/'alias').symlink_to(source,target_is_directory=True)
            with self.assertRaises(ValueError):regular_copy_tree(root/'alias',root/'target')
            (source/'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):regular_copy_tree(source,root/'other')


if __name__=='__main__':
    unittest.main()
