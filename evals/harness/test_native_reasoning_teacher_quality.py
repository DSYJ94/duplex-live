import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evals import native_reasoning_teacher_quality as quality


class WholeTeacherQuality(unittest.TestCase):
    def fixture(self,root,count):
        source=root/'source';source.mkdir();(source/'source-audio.json').write_text('{}')
        q=root/'source-quality.json';q.write_text('{}');batch=root/'batch';batch.mkdir()
        ids=['scene-'+str(i).zfill(2) for i in range(32)];manifest=dict(scenes=[dict(id=i) for i in ids])
        value=dict(schema=quality.teacher.SCHEMA,status='raw_complete_unapproved',source_manifest_sha256=quality.sha(source/'source-audio.json'),
            source_quality_sha256=quality.sha(q),code_hashes=quality.teacher.code_hashes(),attempted_ids=ids[:count],
            recordings=[dict(id=i) for i in ids[:count]],maximum_sessions=32,maximum_concurrent_sessions=2,maximum_backend_requests=192,
            automatic_retry=False,training_approved=False)
        (batch/'teacher-collection.json').write_text(json.dumps(value));return source,q,batch,manifest

    def test_passing31_never_starts_a_whole32_audio_audit(self):
        with tempfile.TemporaryDirectory()as folder:
            source,q,batch,manifest=self.fixture(Path(folder),31)
            with patch.object(quality.source_quality,'verify'),patch.object(quality.teacher,'source_manifest',return_value=(manifest,{})),patch.object(quality.teacher,'geometry'),patch.object(quality.teacher,'verify_record')as readback:
                with self.assertRaisesRegex(ValueError,'complete actual32'):quality.whole_collection(source,q,batch,Path('engine'),Path('spm'))
                readback.assert_not_called()

    def test_each_of_all32_requires_independent_raw_readback(self):
        with tempfile.TemporaryDirectory()as folder:
            source,q,batch,manifest=self.fixture(Path(folder),32)
            with patch.object(quality.source_quality,'verify'),patch.object(quality.teacher,'source_manifest',return_value=(manifest,{})),patch.object(quality.teacher,'geometry'),patch.object(quality.teacher,'verify_record')as readback:
                quality.whole_collection(source,q,batch,Path('engine'),Path('spm'));self.assertEqual(readback.call_count,32)
                readback.side_effect=ValueError('actual provider evidence failed')
                with self.assertRaisesRegex(ValueError,'actual provider evidence'):quality.whole_collection(source,q,batch,Path('engine'),Path('spm'))


if __name__=='__main__':unittest.main()
