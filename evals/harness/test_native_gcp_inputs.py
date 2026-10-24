import hashlib
from pathlib import Path
import tempfile
import subprocess
import unittest

from portable.context.native_gcp_inputs import regular_link, regular_copy_tree, clone_bundle, verify_copy_tree


class InputMaterializationTests(unittest.TestCase):
    def test_head_only_git_bundle_clones_exact_sparse_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source'
            def run(*words):return subprocess.check_output(['git',*map(str,words)],text=True,stderr=subprocess.DEVNULL).strip()
            run('init',source)
            run('-C',source,'config','user.name','Fixture')
            run('-C',source,'config','user.email','fixture@example.invalid')
            for name in ('contextspan/a.py','docs/a.md','omitted/a.txt'):
                file=source/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_text('fixture')
            run('-C',source,'add','.')
            run('-C',source,'commit','-qm','fixture')
            commit=run('-C',source,'rev-parse','HEAD');bundle=root/'source.bundle'
            run('-C',source,'bundle','create',bundle,'HEAD')
            clone_bundle(bundle,root/'clone',commit,('contextspan','docs'))
            self.assertTrue((root/'clone/contextspan/a.py').is_file())
            self.assertFalse((root/'clone/omitted/a.txt').exists())
            self.assertEqual(run('-C',root/'clone','rev-parse','HEAD'),commit)

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
            self.assertEqual(verify_copy_tree(source,target),1)
            (target/'teacher.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'exact verifier'):
                verify_copy_tree(source,target)
            with self.assertRaises(FileExistsError):regular_copy_tree(source,target)

    def test_recovery_rejects_subset_and_shared_inode_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            (source/'receipt.json').write_text('{}');target=root/'target';target.mkdir()
            with self.assertRaisesRegex(ValueError,'Complete'):
                verify_copy_tree(source,target)
            regular_link(source/'receipt.json',target/'receipt.json')
            with self.assertRaisesRegex(ValueError,'Independent'):
                verify_copy_tree(source,target)

    def test_temporary_symlink_roots_and_members_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();source=root/'source';source.mkdir()
            (root/'alias').symlink_to(source,target_is_directory=True)
            with self.assertRaises(ValueError):regular_copy_tree(root/'alias',root/'target')
            (source/'link').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError):regular_copy_tree(source,root/'other')


if __name__=='__main__':
    unittest.main()
