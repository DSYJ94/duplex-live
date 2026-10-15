import json
import tempfile
import unittest
from pathlib import Path

from evals.lychee_utterance_export import export
from evals.moshi_model_download import sha256


class ConcurrentUtteranceExportTest(unittest.TestCase):
    def fixture(self, root):
        import numpy as np
        import soundfile as sf
        source=root/'capture'/'vocoder';source.mkdir(parents=True)
        pcm=np.array([0,.25,-.5,.75],dtype='float32')
        sf.write(source/'packet-stream.wav',np.tile(pcm,2),24000,subtype='PCM_16')
        # First utterance is interrupted after two samples; second is complete.
        sf.write(source/'model.wav',np.concatenate([pcm[:2],np.zeros(2),pcm]),24000,subtype='PCM_16')
        details={f'evt-{i}':{'actual_stop_s':1 if i==1 else None,
            'within_utterance_underrun_samples':0,'source_samples':4,'played_samples':2 if i==1 else 4,
            'packets':[{'playback_start_sample':0 if i==1 else 4}]} for i in (1,2)}
        report={'schema':'dl-lychee-concurrent-pcm-v1','actual_concurrent_lm_execution':True,
            'playout':{'sample_accounting_exact':True,'events':details},'audio_sha256':sha256(source/'model.wav')}
        (source/'report.json').write_text(json.dumps(report))
        (source/'pcm-packets.jsonl').write_text(''.join(json.dumps({'event_id':f'evt-{i}','samples':4})+'\n' for i in (1,2)))
        (source/'utterance-events.json').write_text(json.dumps([{'event_id':f'evt-{i}',
            'text':'The marker is yellow circle.','native_interrupt':i==1,
            'source_ended_by_input_exhaustion':False} for i in (1,2)]))
        return source

    def test_interrupted_text_is_excluded_and_complete_clip_equals_actual_playout(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=self.fixture(root);out=root/'export'
            export(root/'capture',out)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual([Path(p).name for p in manifest['accepted']],['evt-2'])
            self.assertEqual(manifest['excluded'][0]['event']['event_id'],'evt-1')
            played,_=sf.read(source/'model.wav');clip,_=sf.read(out/'evt-2'/'model.wav')
            self.assertTrue((clip==played[4:]).all())
            self.assertFalse(json.loads((out/'evt-2'/'report.json').read_text())['task_success_verified'])

    def test_missing_source_samples_fail_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=self.fixture(root)
            with (source/'pcm-packets.jsonl').open('a') as file:
                file.write(json.dumps({'event_id':'evt-2','samples':1})+'\n')
            with self.assertRaisesRegex(RuntimeError,'cover all source samples'):
                export(root/'capture',root/'export')


if __name__=='__main__':unittest.main()
