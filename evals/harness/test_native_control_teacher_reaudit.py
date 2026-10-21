"""Regression for real transcript-delta shape; no API/model quality claims."""
import unittest
from evals.native_control_teacher_reaudit import transcript_text


class TranscriptBoundaries(unittest.TestCase):
    def test_separate_replies_are_not_glued_and_raw_deltas_remain_recorded(self):
        rows = [dict(speaker='assistant', text='open the', start_ms=9400, end_ms=9600),
                dict(speaker='assistant', text=' small', start_ms=9600, end_ms=9800),
                dict(speaker='assistant', text=' case', start_ms=9800, end_ms=10000),
                dict(speaker='user', text='Walnut', start_ms=25400, end_ms=25600),
                dict(speaker='assistant', text='place', start_ms=26800, end_ms=27000),
                dict(speaker='assistant', text=' the coin inside it', start_ms=27000, end_ms=27200)]
        text, evidence = transcript_text(rows)
        self.assertEqual(text, 'open the small case place the coin inside it')
        self.assertEqual(evidence['raw_concatenation'], 'open the small caseplace the coin inside it')
        self.assertEqual(evidence['inserted_whitespace'][0]['gap_ms'], 16800)
        self.assertFalse(evidence['reference_text_used'])

    def test_contiguous_token_fragments_and_word_differences_are_never_rewritten(self):
        rows = [dict(speaker='assistant', text='app', start_ms=0, end_ms=200),
                dict(speaker='assistant', text='les', start_ms=200, end_ms=400)]
        self.assertEqual(transcript_text(rows)[0], 'apples')
        rows[1]['text'] = 'le'
        self.assertEqual(transcript_text(rows)[0], 'apple')
        self.assertNotEqual(transcript_text(rows)[0], 'apples')

    def test_invalid_original_protocol_times_fail_closed(self):
        for start, end in ((None, 200), (100, 0), (float('nan'), 200)):
            with self.assertRaises(ValueError):
                transcript_text([dict(speaker='assistant', text='word', start_ms=start, end_ms=end)])


if __name__ == '__main__':
    unittest.main()
