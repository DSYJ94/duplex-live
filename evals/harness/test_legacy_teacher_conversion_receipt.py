"""Canonical old quality receipts remain mandatory; V2-only detail stays V2-only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evals.recovery_teacher_convert import validate_teacher_conversion


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class LegacyTeacherConversion(unittest.TestCase):
    def fixture(self,root,modern=False):
        raw=root/'raw';raw.mkdir();(raw/'teacher.json').write_text('{}')
        checks=root/'checks';checks.mkdir();report=checks/'report.json';report.write_text('{"results":[]}')
        count=24 if modern else 0
        approval=dict(speech_report_sha256=sha(report),results=[dict(scenario_id='s'+str(i),speech_and_fact_checks_pass=True) for i in range(count)])
        check=checks/'teacher_checks.json';check.write_text(json.dumps(approval))
        data=root/'converted';data.mkdir();(data/'quality.jsonl').write_text('canonical verified stamp\n')
        (data/'specifications.json').write_text('{}')
        for fold in ('train','validation','test'):(data/(fold+'.jsonl')).write_text('')
        receipt=dict(raw_batch=str(raw),teacher_checks=str(check),teacher_checks_sha256=sha(check),
                     raw_manifest_sha256=sha(raw/'teacher.json'),source_quality_sha256=sha(data/'quality.jsonl'),
                     sealed_specs_sha256=sha(data/'specifications.json'),
                     accepted=[dict(scenario_id='teacher-s'+str(i),split='train') for i in range(count)])
        if modern:
            semantic=root/'semantic.json';semantic.write_text('{}')
            receipt.update(semantic_approval=str(semantic),semantic_approval_sha256=sha(semantic),
                           question_clock_policy='independently-measured-complete-user-words-v1')
        return data,receipt

    def test_legacy_uses_its_original_hash_bound_quality_stamp_without_v2_intermediate(self):
        with tempfile.TemporaryDirectory() as directory:
            data,receipt=self.fixture(Path(directory))
            with patch('evals.recovery_teacher_convert.validate_teacher',return_value={}), \
                 patch('data.synthetic.quality.checked_rows') as canonical:
                self.assertEqual(validate_teacher_conversion(data,receipt),receipt)
                canonical.assert_called_once_with(data,[],role='measured TRAIN-only teacher conversion')
                (data/'quality.jsonl').write_text('changed')
                with self.assertRaisesRegex(RuntimeError,'conversion approval changed'):
                    validate_teacher_conversion(data,receipt)

    def test_new24_corpus_still_requires_full_intermediate_word_quality_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            data,receipt=self.fixture(Path(directory),modern=True)
            with patch('evals.recovery_teacher_convert.validate_teacher',return_value={'collection_schema':'dl-native-dialogue-teacher-v2'}), \
                 patch('evals.native_v2_semantic_review.verify'),patch('data.synthetic.quality.checked_rows'):
                with self.assertRaises(FileNotFoundError):validate_teacher_conversion(data,receipt)


if __name__=='__main__':unittest.main()
