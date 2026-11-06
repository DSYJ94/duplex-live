"""Small stdlib regressions also runnable on Isambard's system Python 3.6."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from training.venus.isambard_artifact_verify import sha,verify


class ArtifactVerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);(self.root/'data').write_bytes(b'bytes')
        (self.root/'link').symlink_to('/original/absolute/path')
        self.inv=dict(files=[dict(path='data',bytes=5,sha256=hashlib.sha256(b'bytes').hexdigest())],links=[dict(path='link',target='/original/absolute/path')],bytes=5)
        self.path=self.root/'inventory.json'

    def run_verify(self):
        self.path.write_text(json.dumps(self.inv))
        return verify(self.root,self.path,sha(self.path))

    def test_all_bytes_and_original_link(self):
        result=self.run_verify()
        self.assertTrue(result['all_hashes_verified']);self.assertEqual(result['links'],1)

    def test_corruption_rejected(self):
        (self.root/'data').write_bytes(b'wrong')
        with self.assertRaises(ValueError):self.run_verify()

    def test_parent_symlink_escape_rejected(self):
        (self.root/'outside').symlink_to('/tmp')
        self.inv['files'][0]['path']='outside/data'
        with self.assertRaisesRegex(ValueError,'parent escapes'):self.run_verify()

    def test_changed_original_link_rejected(self):
        (self.root/'link').unlink();(self.root/'link').symlink_to('/different')
        with self.assertRaisesRegex(ValueError,'link differs'):self.run_verify()


if __name__=='__main__':unittest.main()
