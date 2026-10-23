"""Local copy fixtures; no provider, CUDA, model load, or training approval."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from portable.context.prepare_control40_replacement import safe_name,copy_verified,copy_rows,metadata


class ReplacementInputCopies(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.archive=self.root/'preserved';self.archive.mkdir()
        self.stage=self.root/'fresh';self.stage.mkdir()
        self.name=Path('runs/synthetic-history/voice.pt');source=self.archive/self.name
        source.parent.mkdir(parents=True);source.write_bytes(b'original fixture bytes')
        self.row=dict(path=str(self.name),bytes=source.stat().st_size,sha256=hashlib.sha256(source.read_bytes()).hexdigest())

    def test_copy_and_independent_readback_preserve_original_bytes(self):
        result=copy_rows([self.row],self.archive,self.stage)
        self.assertEqual(result['files'],1);self.assertTrue(result['all_independent_readback_hashes_match'])
        self.assertEqual((self.archive/self.name).read_bytes(),b'original fixture bytes')
        self.assertEqual((self.stage/self.name).read_bytes(),b'original fixture bytes')
        self.assertFalse((self.stage/self.name).with_suffix('.pt.partial').exists())

    def test_changed_source_cannot_publish_a_verified_file(self):
        (self.archive/self.name).write_bytes(b'x'*self.row['bytes'])
        with self.assertRaises(ValueError):copy_rows([self.row],self.archive,self.stage)
        self.assertFalse((self.stage/self.name).exists())
        self.assertTrue((self.archive/self.name).is_file())

    def test_duplicate_or_existing_destination_is_rejected_before_copy(self):
        with self.assertRaises(ValueError):copy_rows([self.row,self.row],self.archive,self.stage)
        self.assertFalse((self.stage/self.name).exists())
        target=self.stage/self.name;target.parent.mkdir(parents=True);target.write_bytes(b'unrelated work')
        with self.assertRaises(ValueError):copy_rows([self.row],self.archive,self.stage)
        self.assertEqual(target.read_bytes(),b'unrelated work')

    def test_source_file_alias_or_destination_parent_alias_rejected(self):
        source=self.archive/self.name;source.unlink();source.symlink_to(self.root/'elsewhere')
        with self.assertRaises(ValueError):copy_rows([self.row],self.archive,self.stage)
        source.unlink();source.write_bytes(b'original fixture bytes')
        for target in (self.archive,self.stage/'another'):
            (self.stage/'runs').symlink_to(target,target_is_directory=True)
            with self.assertRaises(ValueError):copy_rows([self.row],self.archive,self.stage)
            (self.stage/'runs').unlink()

    def test_private_paths_and_cores_cannot_enter_staging(self):
        for name in ('/absolute','../escape','','runs/.private/key','runs/keys/key',
                'runs/owner-conversations/a.wav','runs/synthetic/openai-backend.env','runs/synthetic/core.42'):
            with self.subTest(name=name),self.assertRaises(ValueError):safe_name(name)

    def test_late_concurrent_file_cannot_be_overwritten(self):
        actual_link=os.link;target=self.stage/self.name
        def another_writer(source,destination):
            Path(destination).write_bytes(b'unrelated concurrent edit')
            actual_link(source,destination)
        with patch('portable.context.prepare_control40_replacement.os.link',side_effect=another_writer):
            with self.assertRaises(FileExistsError):
                copy_verified(self.archive/self.name,target,self.row['bytes'],self.row['sha256'],self.archive)
        self.assertEqual(target.read_bytes(),b'unrelated concurrent edit')
        self.assertEqual((self.archive/self.name).read_bytes(),b'original fixture bytes')

    def test_deadline_stops_copy_without_publishing_partial_file(self):
        def expired():raise TimeoutError('fixture deadline')
        with self.assertRaises(TimeoutError):copy_rows([self.row],self.archive,self.stage,expired)
        self.assertFalse((self.stage/self.name).exists())

    def test_only_pinned_inventory_bytes_can_reach_metadata_model_reads(self):
        original=self.root/'original.json';workset=self.root/'workset.json'
        original.write_text('{}');workset.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Exact original10390'):
            metadata(self.archive,original,workset)

    def test_public_model_alias_must_resolve_within_preserved_models(self):
        models=self.archive/'models';models.mkdir();model=models/'snapshot.pt'
        model.symlink_to(self.archive/self.name)
        with self.assertRaises(ValueError):
            copy_verified(model,self.stage/'model.pt',self.row['bytes'],self.row['sha256'],models)
        model.unlink();blob=models/'blob.pt';blob.write_bytes(b'original fixture bytes');model.symlink_to(blob)
        copy_verified(model,self.stage/'model.pt',self.row['bytes'],self.row['sha256'],models)
        self.assertEqual((self.stage/'model.pt').read_bytes(),b'original fixture bytes')


if __name__=='__main__':unittest.main()
