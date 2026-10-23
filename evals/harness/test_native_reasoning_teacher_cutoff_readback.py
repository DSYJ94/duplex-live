"""Fake protocol flags exercise readback code; never provider/teacher approval."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from evals import native_reasoning_teacher_v7 as frozen
from evals import native_reasoning_teacher_cutoff_readback as fixed
from evals import native_reasoning_stream_v6 as stream
from evals.harness.test_native_reasoning_teacher import TeacherArtifacts


class ReadbackFailure(TeacherArtifacts):
    implementation=frozen
    capture_stream=staticmethod(stream.capture)

    async def test_original_nameerror_reproduces_and_corrected_readback_preserves_gates(self):
        with tempfile.TemporaryDirectory()as folder:
            source,scene,out,record,_,_,tokenizer=await self.make_record(Path(folder))
            # These flags are deliberately fabricated for an OFFLINE fixture.
            # No fixture output is ever entered into the actual corpus.
            for name in ['environment.json','report.json']:
                p=out/name;v=json.loads(p.read_text());v['actual_provider_transport']=True
                p.write_text(json.dumps(v))
            for name in record['artifacts']:
                p=out/name;record['artifacts'][name]=dict(bytes=p.stat().st_size,sha256=frozen.sha(p))
            manifest=dict(objects=[dict(object_id='one',wav='user.wav',wav_sha256=frozen.sha(source/'user.wav'))])
            with patch.object(frozen,'source_manifest',return_value=(manifest,{})),patch.object(fixed,'source_manifest',return_value=(manifest,{})):
                with self.assertRaisesRegex(NameError,'transcripts'):
                    frozen.verify_record(source,scene,out,record,tokenizer)
                report=fixed.verify_record(source,scene,out,record,tokenizer)
                self.assertTrue(report['transport_complete'])
                events=out/'events.jsonl';rows=[json.loads(l)for l in events.read_text().splitlines()]
                next(r for r in rows if r['type']=='local.session.close.requested')['required_output_quiet_s']=0.
                events.write_text(''.join(json.dumps(r)+'\n'for r in rows))
                record['artifacts']['events.jsonl']=dict(bytes=events.stat().st_size,sha256=frozen.sha(events))
                with self.assertRaisesRegex(ValueError,'not quiescent'):
                    fixed.verify_record(source,scene,out,record,tokenizer)
