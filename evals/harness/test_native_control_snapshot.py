"""Frozen data identity cannot be replaced by a new passing flag or subset."""
import json
from pathlib import Path
import tempfile
import unittest
from evals import native_control_snapshot as snapshot


class NativeControlSnapshot(unittest.TestCase):
    def test_foreign_inventory_is_rejected_before_any_source_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);inventory=root/'inventory.json'
            for value in (dict(passed=True,files=[]),dict(file_count=7797,files=[])):
                inventory.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError,'source-pinned'):
                    snapshot.verify_files(inventory,root/'not-present')

    def test_snapshot_never_authorizes_another_conversion_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'one exact complete conversion'):
                snapshot.verify(Path(tmp),Path(tmp)/'not-present.json')


if __name__=='__main__':unittest.main()
