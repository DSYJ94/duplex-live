"""New native supervision cannot disable old retention or extend before review."""
from types import SimpleNamespace
import unittest
import json
from pathlib import Path
import tempfile

from evals.context_native_v2_teacher_objective import validate_configuration, validate_teacher_onsets
from evals.moshi_model_download import sha256


class NativeV2ConfigurationTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(native_v2_cache='cache', native_v2_approval='input', native_v2_scope='scope',
                      native_v2_preflight='math', actual_temporal_fit=True, actual_mixed_data=['old28'],
                      warm_start_proof='native520', resume_training=None, actual_fit_updates=8,
                      actual_native_teacher_coverage='all-approved', actual_retention_teacher_weight=1.,
                      actual_base_retention_weight=1., actual_sampling='balanced-kind',
                      actual_retention_sampling='balanced-kind', actual_native_teacher_loss='balanced-spoken',
                      actual_response_first_word_weight=16., actual_native_teacher_approval='old25-input',
                      actual_native_teacher_scope='old25-scope', actual_native_teacher_preflight='old25-math')
        values.update(changes)
        return SimpleNamespace(**values)

    def test_old_paths_do_not_require_new_corpus(self):
        self.assertFalse(validate_configuration(SimpleNamespace()))

    def test_complete_bounded_retention_recipe_is_explicit(self):
        self.assertTrue(validate_configuration(self.config()))

    def test_missing_new_approval_fails(self):
        for name in ('native_v2_cache','native_v2_approval','native_v2_scope','native_v2_preflight'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_configuration(self.config(**{name:None}))

    def test_no_retention_omission_or_unreviewed_extension(self):
        for name,value in [('actual_retention_teacher_weight',0.),('actual_base_retention_weight',0.),
                           ('actual_native_teacher_coverage','four-longest'),('actual_fit_updates',128),
                           ('resume_training','old-state'),('actual_native_teacher_approval',None),
                           ('actual_native_teacher_scope',None),('actual_native_teacher_preflight',None),
                           ('actual_retention_sampling','legacy-modulo'),('actual_response_first_word_weight',1.)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_configuration(self.config(**{name:value}))

    def test_physical_onset_guard_rejects_original_fault_without_changing_audio(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);wav=root/'conversation.wav';pcm=np.zeros((48000,2))
            pcm[24000:,0]=.1
            sf.write(wav,pcm,24000)
            digest=sha256(wav)
            (root/'cache.json').write_text(json.dumps(dict(conversion=str(root))))
            (root/'ingestion.json').write_text(json.dumps(dict(accepted=[dict(
                scenario_id='old-fault',path=str(wav),wav_sha256=digest)])))
            side=wav.with_suffix('.json')
            side.write_text(json.dumps(dict(alignments=[['Sure',[.1,1.2],'SPEAKER_MAIN']])))
            teachers=[dict(id='old-fault',cache=str(root))]
            with self.assertRaisesRegex(ValueError,'physically impossible onset'):
                validate_teacher_onsets(teachers)
            self.assertEqual(sha256(wav),digest)
            # Synthetic control: only a physically valid annotation can pass.
            side.write_text(json.dumps(dict(alignments=[['Sure',[1.,1.2],'SPEAKER_MAIN']])))
            self.assertEqual(validate_teacher_onsets(teachers),['old-fault'])
            self.assertEqual(sha256(wav),digest)


if __name__ == '__main__':
    unittest.main()
