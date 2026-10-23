import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evals.harness import test_native_reasoning_teacher_recovery as previous
from evals import native_reasoning_teacher_recovery_v3 as recovery
from evals import native_reasoning_teacher_quality as quality


class RecoveryPhase(previous.RecoveryPhaseTests):
    implementation=recovery
    maximum_new_sessions=27
    maximum_aggregate_voices=35


class PreservedVersions(unittest.TestCase):
    def test_corrected_replacement_reuses_its_actual_v2_recorder_and_new_capture_uses_v3(self):
        self.assertIs(recovery.recording_implementation(dict(recorder_version='v2',reused_in_v3=True)),recovery.parent_recorder)
        self.assertIs(recovery.recording_implementation(dict(recorder_version='v3',reused_in_v3=False)),recovery.revised)
        for version,reused in (('v3',True),('v1',False)):
            with self.assertRaisesRegex(ValueError,'lineage'):recovery.recording_implementation(dict(recorder_version=version,reused_in_v3=reused))

    def test_v3_schema_cannot_skip_full_actual32_verification(self):
        with tempfile.TemporaryDirectory()as directory:
            p=Path(directory);(p/'teacher-collection.json').write_text(json.dumps(dict(schema=recovery.SCHEMA)))
            with patch.object(recovery,'verify',side_effect=ValueError('Actual32 missing'))as verify:
                with self.assertRaisesRegex(ValueError,'Actual32'):quality.whole_collection(p,p,p,p,p)
                verify.assert_called_once_with(p,p,p,p,p)


if __name__=='__main__':unittest.main()
