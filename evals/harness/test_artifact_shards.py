import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from evals.artifact_shards import split,restore


class ShardTransferTest(unittest.TestCase):
    def test_resume_after_missing_piece_and_reject_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'source'; data=bytes(range(256))*35
            source.write_bytes(data); expected=hashlib.sha256(data).hexdigest()
            split(source,root/'pieces',expected,4096)
            piece=root/'pieces/chunk-0001.bin'; original=piece.read_bytes();piece.unlink()
            with self.assertRaises(FileNotFoundError):restore(root/'pieces/manifest.json',root/'target',expected)
            self.assertFalse((root/'target').exists())
            piece.write_bytes(b'x'*len(original))
            with self.assertRaisesRegex(RuntimeError,'corrupted'):restore(root/'pieces/manifest.json',root/'target',expected)
            self.assertFalse((root/'target').exists())
            piece.write_bytes(original);restore(root/'pieces/manifest.json',root/'target',expected)
            self.assertEqual((root/'target').read_bytes(),data)

    def test_external_pin_and_explicit_destination_protect_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.write_bytes(b'approved model')
            expected=hashlib.sha256(source.read_bytes()).hexdigest()
            split(source,root/'pieces',expected,4)
            target=root/'target';target.write_bytes(b'existing different file')
            with self.assertRaisesRegex(RuntimeError,'overwrite'):restore(root/'pieces/manifest.json',target,expected)
            self.assertEqual(target.read_bytes(),b'existing different file')
            with self.assertRaisesRegex(RuntimeError,'unpinned'):restore(root/'pieces/manifest.json',root/'other','0'*64)
            self.assertFalse((root/'other').exists())


if __name__=='__main__':unittest.main()
