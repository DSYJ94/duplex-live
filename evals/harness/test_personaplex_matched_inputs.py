import json
from pathlib import Path
import tempfile
import unittest
from training.personaplex.matched_inputs import component_teacher_caches,sha


class OriginalManifestBindings(unittest.TestCase):
    def test_compact_audit_without_teacher_uses_exact_original_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);path=folder/'actual-future.json'
            cache='/projects/u6si/duplex-live/runs/recovery-codes-original'
            path.write_text(json.dumps(dict(teacher=dict(cache=cache))))
            audit=dict(component=str(folder),manifest_sha256=sha(path),passed=True)
            self.assertEqual(component_teacher_caches([audit]),{cache})

    def test_action_only_component_remains_eligible_without_inventing_teacher(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);path=folder/'action-future.json'
            path.write_text(json.dumps(dict(target_kind='cancel_request')))
            self.assertEqual(component_teacher_caches([dict(component=str(folder),manifest_sha256=sha(path))]),set())

    def test_changed_manifest_is_rejected_before_its_paths_are_used(self):
        with tempfile.TemporaryDirectory() as root:
            folder=Path(root);path=folder/'early-future.json';path.write_text('{}')
            row=dict(component=str(folder),manifest_sha256=sha(path));path.write_text('{"teacher": {}}')
            with self.assertRaisesRegex(ValueError,'manifest changed'):component_teacher_caches([row])


if __name__=='__main__':unittest.main()
