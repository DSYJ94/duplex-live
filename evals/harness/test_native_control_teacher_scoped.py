"""Scoped TRAIN prompt and byte-preserving real recorder seam; no model approval."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
from evals import native_control_teacher_scoped as scoped
from evals import native_control_teacher as original
from evals.harness.test_native_control_teacher import Clock
from evals.harness.test_native_control_teacher_drained import BacklogSocket

class ScopedTeacher(unittest.IsolatedAsyncioTestCase):
    def test_only_instructions_change_and_policy_waits_for_complete_question(self):
        old,new=original.configuration(),scoped.configuration()
        self.assertEqual({k:v for k,v in old.items() if k!='instructions'},
                         {k:v for k,v in new.items() if k!='instructions'})
        self.assertIn('complete final option',new['instructions'])
        self.assertEqual(new['instructions'].count('Reflected light carries light to your eyes.'),1)
        self.assertIn('seven words exactly once',new['instructions'])
        self.assertIn('rest of the recording',new['instructions'])
        self.assertNotIn('Point at the window',new['instructions'])
        self.assertNotIn('Do not answer only Reflected light',new['instructions'])

    async def test_actual_echo_full_raw_pcm_and_postclose_clock_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sid=scoped.FAILED_IDS[0];(root/sid).mkdir()
            sf.write(root/sid/'stem_user.wav',np.full(1920,.1,dtype='float32'),24000,subtype='FLOAT')
            (root/'raw-source.json').write_text('{"fixture_only":true}')
            clock,socket=Clock(),BacklogSocket()
            class Connection:
                async def __aenter__(self):return socket
                async def __aexit__(self,*args):return None
            capture=scoped.capture_stream
            async def run(*args,**kwargs):return await capture(*args,**kwargs,now=clock.now,sleep=clock.sleep)
            folder=root/'record'
            with patch.object(scoped,'capture_stream',side_effect=run):
                record=await scoped.capture_one(root,dict(id=sid,frames=1920),folder,'fixture-audit',
                                               'fixture-key',connector=lambda *a,**kw:Connection())
            self.assertEqual(next(e for e in socket.sent if e['type']=='session.start')['session'],scoped.configuration())
            self.assertEqual((folder/'output-stream.pcm').read_bytes(),socket.output)
            scoped.verify_exact_record(root,dict(id=sid,frames=1920),folder,'fixture-audit',record)
            events=[json.loads(e) for e in (folder/'events.jsonl').read_text().splitlines()]
            next(e for e in events if e['type']=='session.started')['session']['instructions']='wrong fixture instructions'
            (folder/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
            record['artifacts']['events.jsonl']=scoped.sha256(folder/'events.jsonl')
            with self.assertRaisesRegex(ValueError,'Incomplete actual session'):
                scoped.verify_exact_record(root,dict(id=sid,frames=1920),folder,'fixture-audit',record)

    async def test_single_new_session_scope_is_checked_before_data_or_network(self):
        for n in (0,2,True):
            with self.assertRaisesRegex(ValueError,'Exactly one'):
                await scoped.collect(*(Path('absent') for _ in range(6)),'fixture-key',Path('lease'),concurrency=n)

    def test_semantic_and_conversion_remain_after_complete_quality_verifier(self):
        from evals import native_control_teacher_scoped_quality as quality
        from evals import native_control_semantic as semantic
        from evals import native_control_postteacher_pipeline as pipeline
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);q=root/'q.json';q.write_text(json.dumps(dict(schema=quality.SCHEMA)))
            with patch.object(quality,'verify',side_effect=ValueError('Whole40 failed')) as verify:
                with self.assertRaisesRegex(ValueError,'Whole40 failed'):semantic.verify_quality(root,root,root,q)
                verify.assert_called_once_with(root,root,root,q)
        rows=list(pipeline.stages(*(Path('fixture') for _ in range(8)),scoped_teacher=True))
        self.assertEqual(rows[0][1],'evals.native_control_teacher_scoped_quality')
        self.assertEqual(rows[1][1],'evals.native_control_teacher_scoped_quality')
        self.assertIn('--verify',rows[1][2])
        self.assertEqual(rows[2][1],'evals.native_control_semantic')
        with self.assertRaises(ValueError):list(pipeline.stages(*(Path('fixture') for _ in range(8)),scoped_teacher=True,drained_teacher=True))

if __name__=='__main__':unittest.main()
