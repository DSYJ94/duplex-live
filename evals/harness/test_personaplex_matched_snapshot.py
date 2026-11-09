import hashlib,io,tarfile
from pathlib import Path
import tempfile
import unittest
from training.personaplex.matched_snapshot import destination,extract


class OriginalWorksetRestoration(unittest.TestCase):
    def archive(self,path,members):
        rows=[]
        with tarfile.open(path,'w:gz') as f:
            for name,payload in members:
                m=tarfile.TarInfo(name);m.size=len(payload);f.addfile(m,io.BytesIO(payload))
                rows.append(dict(path=name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
        return rows

    def test_every_member_and_second_destination_read_are_verified(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);archive=root/'test.tar.gz'
            rows=self.archive(archive,[('runs/native-control3-test/one.json',b'original'),('runs/native-control3-test/two.wav',b'PCM')])
            out=root/'out';r=extract(archive,rows,out)
            self.assertEqual((r['files'],r['bytes']),(2,11))
            self.assertEqual((out/rows[0]['path']).read_bytes(),b'original')
            with self.assertRaises(FileExistsError):extract(archive,rows,out)

    def test_missing_extra_or_hash_changed_member_never_passes(self):
        for change in ('missing','extra','hash'):
            with tempfile.TemporaryDirectory() as root:
                root=Path(root);archive=root/'test.tar.gz'
                rows=self.archive(archive,[('runs/native-control3-test/one.json',b'original')])
                if change=='missing':rows+=[dict(path='runs/native-control3-test/two.json',bytes=0,sha256=hashlib.sha256(b'').hexdigest())]
                elif change=='extra':rows=[]
                else:rows[0]['sha256']='0'*64
                with self.subTest(change=change),self.assertRaises(ValueError):extract(archive,rows,root/'out')

    def test_paths_cannot_escape_or_include_private_or_unrelated_work(self):
        for name in ('/etc/passwd','runs/../private/key','runs/other/file','runs/native-control3-test/.private/key','runs/native-control3-test/session-token'):
            with self.subTest(name=name),self.assertRaises(ValueError):destination(Path('/tmp/fixture'),name)

    def test_resume_reuses_valid_bytes_and_preserves_failed_partial_copy(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);archive=root/'test.tar.gz'
            rows=self.archive(archive,[('runs/native-control3-test/one.json',b'original'),('runs/native-control3-test/two.wav',b'PCM')])
            out=root/'out';one=out/rows[0]['path'];one.parent.mkdir(parents=True);one.write_bytes(b'original')
            before=one.stat().st_mtime_ns;two=out/rows[1]['path'];two.write_bytes(b'P')
            r=extract(archive,rows,out,resume=True)
            self.assertEqual(r['existing_valid_files_reused'],1)
            self.assertEqual(r['partial_files_preserved_and_repaired'],1)
            self.assertEqual(one.stat().st_mtime_ns,before)
            self.assertEqual(two.read_bytes(),b'PCM')
            self.assertEqual((root/'out-partial-preserved-v1'/rows[1]['path']).read_bytes(),b'P')


if __name__=='__main__':unittest.main()
