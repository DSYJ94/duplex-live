"""New native supervision cannot disable old retention or extend before review."""
from types import SimpleNamespace
import unittest
import json
from pathlib import Path
import tempfile

from evals.context_native_v2_teacher_objective import validate_configuration, validate_teacher_onsets, validate_old25_lineage
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
                      actual_native_teacher_scope='old25-scope', actual_native_teacher_preflight='old25-math',
                      actual_native_teacher_cache='repaired-old25')
        values.update(changes)
        return SimpleNamespace(**values)

    def test_old_paths_do_not_require_new_corpus(self):
        self.assertFalse(validate_configuration(SimpleNamespace()))

    def test_complete_bounded_retention_recipe_is_explicit(self):
        self.assertTrue(validate_configuration(self.config()))

    def test_explicit_full24_pilot_preserves_old_weights_and_requires_review_for_resume(self):
        for rate in (1e-5,2e-6):
            self.assertTrue(validate_configuration(self.config(native_v2_full24_per8=True,native_v2_learning_rate=rate)))
        for changes in [dict(native_v2_learning_rate=2e-6),
                        dict(native_v2_full24_per8=True,native_v2_learning_rate=1e-4),
                        dict(native_v2_full24_per8=True,actual_fit_updates=32,resume_training='state'),
                        dict(native_v2_full24_per8=True,native_v2_gradient_audit=True),
                        dict(native_v2_full24_per8=True,actual_retention_teacher_weight=0.)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_configuration(self.config(**changes))

    def test_missing_new_approval_fails(self):
        for name in ('native_v2_cache','native_v2_approval','native_v2_scope','native_v2_preflight'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_configuration(self.config(**{name:None}))

    def test_no_retention_omission_or_unreviewed_extension(self):
        for name,value in [('actual_retention_teacher_weight',0.),('actual_base_retention_weight',0.),
                           ('actual_native_teacher_coverage','four-longest'),('actual_fit_updates',128),
                           ('resume_training','old-state'),('actual_native_teacher_approval',None),
                           ('actual_native_teacher_scope',None),('actual_native_teacher_preflight',None),
                           ('actual_native_teacher_cache',None),
                           ('actual_retention_sampling','legacy-modulo'),('actual_response_first_word_weight',1.)]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_configuration(self.config(**{name:value}))

    def test_old25_timing_lineage_retains_every_undiagnosed_target_and_retention_weight(self):
        import copy
        from evals.native_old25_timing_repair import REPAIRED
        ids = sorted(REPAIRED) + ['other-'+str(i) for i in range(21)]
        old = dict(weight=1., reference_kl_and_hidden_objective_unchanged=True,
            teachers=[dict(id=i, cache='old', cache_sha256='old', ingestion_sha256='old',
                           targets_sha256='same', wav_sha256='audio', frames=100) for i in ids])
        new = copy.deepcopy(old)
        for row in new['teachers']:
            row.update(cache='new', cache_sha256='new', ingestion_sha256='new')
            if row['id'] in REPAIRED:row['targets_sha256']='measured-new-clock'
        validate_old25_lineage(old,new)
        for change in ('audio','frames','undiagnosed','drop','weight','kl'):
            bad=copy.deepcopy(new)
            if change=='audio':bad['teachers'][0]['wav_sha256']='changed'
            elif change=='frames':bad['teachers'][0]['frames']=99
            elif change=='undiagnosed':bad['teachers'][-1]['targets_sha256']='changed'
            elif change=='drop':bad['teachers'].pop()
            elif change=='weight':bad['weight']=0.
            else:bad['reference_kl_and_hidden_objective_unchanged']=False
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_old25_lineage(old,bad)

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
