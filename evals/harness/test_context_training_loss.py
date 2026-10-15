import unittest
from types import SimpleNamespace
import torch
from evals.context_training_loss import loss_fn


class ContextLossTest(unittest.TestCase):
    def fixture(self):
        codes = torch.ones(1, 17, 4, dtype=torch.long)
        codes[:, :, 0] = -1
        mask = torch.tensor([[False, True, True, True]])
        out = SimpleNamespace(text_logits=torch.zeros(1, 1, 4, 8),
            logits=torch.zeros(1, 16, 4, 8), text_mask=mask[:, None].clone(),
            mask=mask[:, None].expand(1, 16, 4).clone())
        out.text_logits[:, :, 0] = float('nan')
        out.logits[:, :, 0] = float('nan')
        return out, codes, mask, SimpleNamespace(audio_offset=1)

    def test_invalid_and_virtual_labels_excluded_before_ce(self):
        out, codes, mask, lm = self.fixture()
        loss, report = loss_fn(out, codes, mask, mask, mask, lm)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(report['target_counts'], dict(text=3, assistant_audio=24,
            clean_user_audio=24, user_targets_excluded_by_ownership=0))

    def test_user_attribution_preserves_assistant_and_real_overlap(self):
        out, codes, mask, lm = self.fixture()
        ownership = mask.clone(); ownership[:, 2] = False
        _, report = loss_fn(out, codes, mask, mask, ownership, lm)
        self.assertEqual(report['target_counts']['assistant_audio'], 24)
        self.assertEqual(report['target_counts']['clean_user_audio'], 16)
        self.assertEqual(report['target_counts']['user_targets_excluded_by_ownership'], 8)
        self.assertTrue(torch.equal(codes[:, 1:9], codes[:, 9:17]))

    def test_valid_nonfinite_predictions_are_not_hidden(self):
        out, codes, mask, lm = self.fixture()
        out.logits[:, 0, 1, 0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'nonfinite valid audio'):
            loss_fn(out, codes, mask, mask, mask, lm)

    def test_ownership_cannot_enable_prefix(self):
        out, codes, mask, lm = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'virtual/prefix'):
            loss_fn(out, codes, mask, mask, torch.ones_like(mask), lm)


if __name__ == '__main__': unittest.main()
