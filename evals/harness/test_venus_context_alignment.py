import unittest
from data.venus.context_alignment import admission, measured_units


class ContextAlignmentTests(unittest.TestCase):
    def test_complete_original_batch_and_cost_admission(self):
        self.assertEqual(len(admission()), 34)

    def test_compact_numeric_chunk_cannot_supply_script_word_times(self):
        with self.assertRaises(ValueError):
            measured_units('Five hundred pounds.', dict(words=[dict(text='£500', start=0, end=1)]), 2)

    def test_changed_or_missing_words_rejected(self):
        with self.assertRaises(ValueError):
            measured_units('Stay quiet.', dict(words=[dict(text='Speak', start=0, end=1)]), 2)

    def test_unchanged_text_and_measured_word_bins_preserved(self):
        result = measured_units('Five hundred pounds.', dict(words=[dict(text='Five', start=.1, end=.4), dict(text='hundred', start=.5, end=.9), dict(text='pounds.', start=1.1, end=1.6)]), 2)
        self.assertEqual([r['text'] for r in result], ['Five hundred', 'pounds.'])

    def test_timestamp_beyond_original_audio_rejected(self):
        with self.assertRaises(ValueError):
            measured_units('Five.', dict(words=[dict(text='Five.', start=0, end=3)]), 2)
