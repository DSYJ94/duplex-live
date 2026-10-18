import json
from pathlib import Path
import tempfile
import unittest
from evals.duplex_fresh_inputs import checked_cases
from evals.moshi_model_download import sha256
import evals.duplex_fresh_inputs as source


class FreshInputGuards(unittest.TestCase):
    def test_rejects_unjudged_changed_training_or_nonfinite_sources(self):
        # Exercise actual manifest/quality/source binding, no GPU fixture or
        # inference claim. Every mutation must reject the same selected input.
        for mutation in ('none','empty-clips','nan','failed','changed-wav','training','wrong-manifest'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);wav=root/'user.wav';wav.write_bytes(b'actual input bytes')
                ref=root/'reference.json';ref.write_text('{"user_words":[]}')
                row=dict(scenario_id='fresh',split='fresh-evaluation',path=str(wav),reference_path=str(ref),
                    wav_sha256=sha256(wav),reference_sha256=sha256(ref),utterance_count=1)
                m=dict(schema='dl-hybrid-fresh-inputs-v1',rows=[row],training_approved=mutation=='training',do_not_train=True,
                    assistant_audio_generated=False,generator_sha256=sha256(Path(source.__file__)))
                file=root/'inputs.json';file.write_text(json.dumps(m))
                clip=dict(passed=True,wer=0.,valid_bounds=True)
                approval=dict(scenario_id='fresh',passed=True,clips=[clip],wav_sha256=row['wav_sha256'],reference_sha256=row['reference_sha256'])
                quality=dict(schema='dl-hybrid-fresh-quality-v1',inputs_manifest_sha256=sha256(file),training_approved=False,do_not_train=True,
                    threshold=.05,judge_sha256=sha256(Path(source.__file__)),asr_model='whisper turbo',results=[approval])
                if mutation=='empty-clips':approval['clips']=[]
                if mutation=='nan':clip['wer']=float('nan')
                if mutation=='failed':approval['passed']=False
                if mutation=='changed-wav':wav.write_bytes(b'different')
                if mutation=='wrong-manifest':quality['inputs_manifest_sha256']='changed'
                q=root/'quality.json';q.write_text(json.dumps(quality))
                if mutation=='none':self.assertEqual(checked_cases(root,q,['fresh']),[row])
                else:
                    with self.assertRaises(RuntimeError):checked_cases(root,q,['fresh'])


if __name__=='__main__':unittest.main()
