import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from portable.context.native_gcp_stage import control_archive, materialize_files


class StageTests(unittest.TestCase):
    def archive(self, root, specs):
        path = root/'input.tar.gz'
        with tarfile.open(path, 'w:gz') as writer:
            for name, kind in specs:
                info = tarfile.TarInfo(name)
                if kind == 'file':
                    info.size = 4
                    writer.addfile(info, io.BytesIO(b'data'))
                else:
                    info.type = tarfile.SYMTYPE
                    info.linkname = '/etc/passwd'
                    writer.addfile(info)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def row(self):
        return dict(path='runs/a', bytes=4, sha256=hashlib.sha256(b'data').hexdigest())

    def test_complete_regular_archive_is_restored_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); archive, sha = self.archive(root, [('runs/a', 'file')])
            self.assertEqual(control_archive(archive, root/'out', [self.row()], sha), 1)
            self.assertEqual((root/'out/runs/a').read_bytes(), b'data')
            with self.assertRaises(FileExistsError):
                control_archive(archive, root/'out', [self.row()], sha)

    def test_unexpected_duplicate_traversal_and_symlink_members_fail(self):
        for specs in ([('unexpected', 'file')], [('runs/a', 'file'), ('runs/a', 'file')],
                      [('../outside', 'file')], [('runs/a', 'symlink')], []):
            with self.subTest(specs=specs), tempfile.TemporaryDirectory() as d:
                root = Path(d); archive, sha = self.archive(root, specs)
                with self.assertRaises(ValueError):
                    control_archive(archive, root/'out', [self.row()], sha)
                self.assertFalse((root/'outside').exists())

    def test_changed_container_or_member_checksum_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); archive, sha = self.archive(root, [('runs/a', 'file')])
            with self.assertRaisesRegex(ValueError, 'archive SHA'):
                control_archive(archive, root/'bad', [self.row()], '0'*64)
            row = dict(self.row(), sha256='0'*64)
            with self.assertRaisesRegex(ValueError, 'member SHA'):
                control_archive(archive, root/'out', [row], sha)

    def test_catalogued_file_is_hardlinked_but_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); transport = root/'transport'; (transport/'files/runs').mkdir(parents=True)
            (transport/'files/runs/a').write_bytes(b'data')
            materialize_files(transport, root, [self.row()])
            self.assertEqual((root/'runs/a').stat().st_ino, (transport/'files/runs/a').stat().st_ino)
            with self.assertRaises(FileExistsError):materialize_files(transport, root, [self.row()])


if __name__ == '__main__':
    unittest.main()
