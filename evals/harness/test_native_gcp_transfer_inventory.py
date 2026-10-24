"""Migration-scope and checksum fixtures; no transfer, Torch or GPU work."""
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.native_gcp_transfer_inventory import safe,rows,unique,load


class TransferInventoryTests(unittest.TestCase):
    def test_traversal_credentials_owner_conversations_and_dump_paths_are_rejected(self):
        for name in ('../outside','/absolute','a/../outside','./a','a//b','.',
                'x/.private/key','x/.PRIVATE/key','x/.ssh/key','x/credentials/value',
                'x/owner-conversations/audio.wav','x/session-token','x/core','x/core.1234',
                'x/core-dump','x\nother','x\x00other','x\\other'):
            with self.subTest(path=name),self.assertRaises(ValueError):safe(name)
        self.assertEqual(safe('native-preservation-v3/envs/a/core/tensor.py'),
            'native-preservation-v3/envs/a/core/tensor.py')

    def test_paths_are_unique_sorted_and_every_file_has_sha256(self):
        row=dict(path='migration/file.pt',bytes=23,sha256='a'*64)
        self.assertEqual(rows([row]),23)
        for altered in ([row,row],[dict(row,sha256='s3-etag')],[dict(row,bytes=True)],
                [dict(row,bytes=-1)],[dict(row,local_workspace_path='../private')],
                [dict(row,credential='not allowed')]):
            with self.assertRaises(ValueError):rows(altered)
        with self.assertRaises(ValueError):rows([dict(row,path='z'),dict(row,path='a')])

    def test_proposed_byte_limit_is_enforced_without_reading_payloads(self):
        with self.assertRaises(ValueError):rows([dict(path='migration/large.pt',bytes=100000000001,sha256='a'*64)])

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):unique([('path','a'),('path','b')])

    def test_compressed_bytes_are_pinned_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'inventory.gz';p.write_bytes(gzip.compress(b'{}',mtime=0))
            with self.assertRaisesRegex(ValueError,'inventory bytes changed'):load(p,'0'*64)

    def test_decompression_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'inventory.gz';data=gzip.compress(b'x'*100,mtime=0);p.write_bytes(data)
            with patch('evals.native_gcp_transfer_inventory.MAX_JSON_BYTES',16),self.assertRaisesRegex(ValueError,'Bounded inventory'):
                load(p,hashlib.sha256(data).hexdigest())

    def test_actual_closed_catalog_is_complete_and_never_an_execution_permit(self):
        p=Path(__file__).resolve().parents[2]/'docs/GCP-NATIVE-TRANSFER-INVENTORY-2026-10-24.json.gz'
        value,report=load(p,'94b42bae0152e8150ba7842146f66711b13df3b8ece9125bcb5db92a761196fb')
        self.assertEqual(report['regular_files'],40132)
        self.assertEqual(report['expanded_artifact_bytes'],81154452891)
        self.assertEqual(report['archived_symlinks'],8)
        self.assertEqual(value['archive_transport_bytes'],74228899463)
        self.assertFalse(report['transfer_or_compute_authorized'])
        self.assertFalse(report['destination_verified'])
        self.assertFalse(report['model_adopted'])


if __name__=='__main__':unittest.main()
