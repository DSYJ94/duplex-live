import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from training.personaplex import matched_archive as a


class ArchiveIntegrity(unittest.TestCase):
    def test_failed_archive_is_recorded_without_changing_training_terminal(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);ex=root/'execution-999';ex.mkdir();terminal=ex/'terminal.json'
            terminal.write_text('{"actual_exit_code":0,"source":"worker"}')
            child=SimpleNamespace(returncode=1,wait=lambda **kw:1)
            a.finish_helper(child,root,'999','worker')
            self.assertEqual(json.loads(terminal.read_text())['actual_exit_code'],0)
            record=json.loads((ex/'independent-archive-terminal.json').read_text())
            self.assertEqual(record['actual_exit_code'],1);self.assertFalse(record['training_interrupted'])

    def fixture(self,root):
        state=root/'epoch-02.pt';state.write_bytes(b'complete synthetic state fixture')
        receipt=root/'epoch-02.json';value=dict(source='actual-source',lineage_updates=98,checkpoint_sha256=a.sha(state));receipt.write_text(json.dumps(value))
        Path(str(state)+'.native-reference.json').write_text(json.dumps(dict(source='actual-source',step=98,checkpoint_sha256=value['checkpoint_sha256'])))
        return state,receipt,root/'destination'

    def test_corrupt_source_or_wrong_reference_never_starts_transfer(self):
        with tempfile.TemporaryDirectory() as d:
            state,receipt,destination=self.fixture(Path(d));state.write_bytes(b'truncated')
            with patch.object(a.subprocess,'run') as transfer:
                with self.assertRaises(ValueError):a.archive(state,receipt,destination,'actual-source')
                transfer.assert_not_called()
        with tempfile.TemporaryDirectory() as d:
            state,receipt,destination=self.fixture(Path(d));ref=Path(str(state)+'.native-reference.json');value=json.loads(ref.read_text());value['step']=97;ref.write_text(json.dumps(value))
            with patch.object(a.subprocess,'run') as transfer:
                with self.assertRaises(ValueError):a.archive(state,receipt,destination,'actual-source')
                transfer.assert_not_called()

    def test_existing_backup_corruption_cannot_be_called_verified(self):
        with tempfile.TemporaryDirectory() as d:
            state,receipt,destination=self.fixture(Path(d));destination.mkdir();(destination/state.name).write_bytes(b'bad copy')
            (destination/'archive-receipt.json').write_text(json.dumps(dict(sha256=a.sha(state),source='actual-source',step=98)))
            # Skip only placement in this same-filesystem local corruption fixture.
            with patch.object(a,'independent_filesystems',return_value=True),self.assertRaisesRegex(ValueError,'never overwrite'):a.archive(state,receipt,destination,'actual-source')


if __name__=='__main__':unittest.main()
