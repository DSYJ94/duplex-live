import unittest
from evals.lychee_loss_scope import check


class LossOwnershipTest(unittest.TestCase):
    def fixture(self):
        # Identical user/assistant token IDs do not identify an echo. Labels,
        # actual history boundary and verified assistant text define ownership.
        return dict(input_ids=[1,7,8,9,7,8,9,2],labels=[-100]*4+[7,8,9]+[-100],
            expected=[7,8,9],history_end=4,system_length=1,received_at_s=.10,
            declared_first_s=.12,declared_count=3)

    def test_preserves_context_and_assistant_targets_during_overlap(self):
        args=self.fixture();before=args['input_ids'][:]
        result=check(**args)
        self.assertEqual(args['input_ids'],before)
        self.assertEqual(result['assistant_text_targets'],3)
        self.assertEqual(result['user_prediction_targets'],0)
        self.assertFalse(result['aec_applied'])

    def test_prompt_or_user_history_cannot_become_assistant_target(self):
        args=self.fixture();args['labels'][1]=7
        with self.assertRaisesRegex(RuntimeError,'history'):check(**args)

    def test_audio_or_foreign_text_cannot_replace_verified_assistant_target(self):
        args=self.fixture();args['labels'][4]=151700;args['input_ids'][4]=151700
        with self.assertRaisesRegex(RuntimeError,'independently sourced'):check(**args)

    def test_causal_clock_uses_target_position_before_loss_shift(self):
        args=self.fixture();args['received_at_s']=.121
        with self.assertRaisesRegex(RuntimeError,'precedes actual receipt'):check(**args)


if __name__=='__main__':unittest.main()
