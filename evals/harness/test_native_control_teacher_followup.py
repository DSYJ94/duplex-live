"""Diagnosed followup inventory guards; fixtures do not approve any real label."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals import native_control_teacher_followup as followup


class FollowupInventory(unittest.TestCase):
    def test_only_two_diagnosed_records_can_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'teacher.json').write_text('{}')
            quality, diagnosis = root / 'quality.json', root / 'diagnosis.json'
            quality.write_text('{}')
            diagnosis.write_text('{}')
            records = [dict(id=f'native-control3-train-{i:03d}', artifacts=dict(wave='fixture')) for i in range(40)]
            parent = dict(recordings=records, repair=dict(fixture_only=True))
            retained = [r['id'] for r in records if r['id'] not in followup.FAILED_IDS]
            lineage = dict(schema='dl-control3-diagnosed-two-teacher-followup-v1',
                           code_sha256=followup.sha256(Path(followup.__file__)), parent_batch=str(root),
                           parent_quality=str(quality), diagnosis=str(diagnosis),
                           parent_batch_sha256=followup.sha256(root / 'teacher.json'),
                           parent_quality_sha256=followup.sha256(quality), diagnosis_sha256=followup.sha256(diagnosis),
                           maximum_new_sessions=2, new_session_ids=followup.FAILED_IDS,
                           retained_records=38, retained_ids=retained, inputs_or_config_or_recorder_changed=False)
            manifest = copy.deepcopy(dict(parent, repair_followup=lineage))
            for row in manifest['recordings']:
                if row['id'] in followup.FAILED_IDS:
                    row['artifacts']['wave'] = 'new-fixture'
            with patch.object(followup, 'evidence', return_value=parent):
                followup.verify_followup(root, quality, root, manifest)
                bad = copy.deepcopy(manifest)
                bad['recordings'][0]['artifacts']['wave'] = 'changed-healthy-fixture'
                with self.assertRaisesRegex(ValueError, 'passing original'):
                    followup.verify_followup(root, quality, root, bad)
                bad = copy.deepcopy(manifest)
                bad['repair_followup']['maximum_new_sessions'] = 3
                with self.assertRaisesRegex(ValueError, 'scope/receipt'):
                    followup.verify_followup(root, quality, root, bad)


if __name__ == '__main__':
    unittest.main()
