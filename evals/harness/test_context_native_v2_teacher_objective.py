"""New native supervision cannot disable old retention or extend before review."""
from types import SimpleNamespace
import unittest

from evals.context_native_v2_teacher_objective import validate_configuration


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


if __name__ == '__main__':
    unittest.main()
