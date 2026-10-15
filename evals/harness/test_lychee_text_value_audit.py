import unittest
from evals.lychee_text_value_audit import inspect_snapshots


class LiteralTimingTests(unittest.TestCase):
    def test_private_word_started_before_receipt_does_not_pass_when_completed_later(self):
        rows=[{'event_id':'one','text':'The marker is yel','at_s':1.},
            {'event_id':'one','text':'The marker is yellow','at_s':2.},
            {'event_id':'one','text':'The marker is yellow circle.','at_s':3.}]
        r=inspect_snapshots(rows,'yellow circle',2.5)
        self.assertFalse(r['actual_value_mentioned_after_delivery'])
        self.assertTrue(r['any_colour_shape_started_before_delivery'])
        self.assertEqual(r['literal_mentions'][0]['first_word_token_at_s'],1.)

    def test_literal_echo_is_not_promoted_to_task_or_speech_success(self):
        r=inspect_snapshots([{'event_id':'one','text':"It is not blue square; it is yellow circle.",'at_s':3.}],
            'yellow circle',2.)
        self.assertTrue(r['actual_value_mentioned_after_delivery'])
        self.assertEqual(len(r['other_colour_shape_mentions']),1)
        self.assertFalse(r['task_success_verified'])
        self.assertFalse(r['spoken_result_use_verified'])


if __name__=='__main__':unittest.main()
