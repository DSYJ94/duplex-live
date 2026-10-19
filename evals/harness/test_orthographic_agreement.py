import copy
import unittest
from evals.recovery_prepare_data import agreement_wer, orthographic_agreement, spoken_words


class OrthographicAgreement(unittest.TestCase):
    def record(self, model, asr):
        score=orthographic_agreement(model,asr)
        return dict(model_text=model, asr_text=asr, wer_against_model_text=score['raw_norm_wer'],
                    word_agreement=score)

    def test_same_spoken_inflections_retain_raw_error_and_explicit_canonical_zero(self):
        r=self.record('Cancelling the lookup. The lookup is cancelled.',
                      'Canceling the lookup. The lookup is canceled.')
        self.assertGreater(r['wer_against_model_text'], .08)
        self.assertEqual(agreement_wer(r),0.)
        self.assertEqual(spoken_words(r['model_text']),spoken_words(r['asr_text']))
        del r['word_agreement']
        self.assertGreater(agreement_wer(r),.08)  # Legacy raw diagnostic remains unchanged.

    def test_no_missing_repeated_substituted_words_or_synonyms_are_accepted(self):
        for model,asr in [('Go.','Go. Go.'), ('A timer measures time.','Timer measures time.'),
                          ('The bus arrived first.','The boss arrived first.'),
                          ('Roots take in water.','Roots absorb water.'),
                          ('The lookup is cancelled.','The lookup is complete.')]:
            with self.subTest(asr=asr):
                r=self.record(model,asr)
                self.assertGreater(agreement_wer(r),.08)
                self.assertNotEqual(spoken_words(model),spoken_words(asr))

    def test_modified_raw_score_or_declared_policy_is_rejected(self):
        r=self.record('Cancelling.','Canceling.')
        for field in ('raw','canonical','variants'):
            bad=copy.deepcopy(r)
            if field=='raw':bad['wer_against_model_text']=0.
            elif field=='canonical':bad['word_agreement']['canonical_word_wer']=1.
            else:bad['word_agreement']['variants']['boss']='bus'
            with self.subTest(field=field), self.assertRaises(RuntimeError):agreement_wer(bad)

    def test_conversion_quality_stamp_uses_same_policy_and_preserves_raw_score(self):
        from evals.recovery_teacher_convert import measured_speech_quality
        r=self.record('Cancelling the lookup. The lookup is cancelled.',
                      'Canceling the lookup. The lookup is canceled.')
        row=measured_speech_quality(r)
        self.assertEqual(row['wer'],0.)
        self.assertGreater(row['raw_wer_against_model_text'],.08)
        self.assertEqual(row['word_agreement'],r['word_agreement'])
        del r['word_agreement']
        self.assertGreater(measured_speech_quality(r)['wer'],.08)


if __name__=='__main__':unittest.main()
