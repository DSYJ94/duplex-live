"""Exercise real LMGen queue/consume methods without speech weights or a GPU."""
import unittest
from types import MethodType, SimpleNamespace

import torch
from moshi.models import LMGen
from evals.moshi_rag_reference_step import apply_reference_frame


class ReferenceStepTest(unittest.TestCase):
    def test_delivery_alone_is_not_consumption_and_rows_clear(self):
        state = SimpleNamespace(batch_size=1, pending_streaming_sums=[None],
                                condition_streaming_sum=torch.zeros(1, 1, 4))
        generation = SimpleNamespace(cfg_coef=1.0, _streaming_state=state,
                                      lm_model=SimpleNamespace(device="cpu", dtype=torch.float32))
        generation.apply_pending_streaming_sum_condition = MethodType(
            LMGen.apply_pending_streaming_sum_condition, generation)
        LMGen.update_streaming_sum_tensors(generation, [torch.stack([torch.ones(4), torch.full((4,), 2.)])])
        self.assertEqual(float(state.condition_streaming_sum.abs().sum()), 0.)
        self.assertEqual(apply_reference_frame(generation), 4.)
        self.assertEqual(state.pending_streaming_sums[0].shape, (1, 4))
        self.assertEqual(apply_reference_frame(generation), 8.)
        self.assertIsNone(state.pending_streaming_sums[0])
        self.assertEqual(apply_reference_frame(generation), 0.)

    def test_omitted_application_is_detected(self):
        state = SimpleNamespace(batch_size=1, pending_streaming_sums=[torch.ones(2, 4)],
                                condition_streaming_sum=torch.zeros(1, 1, 4))
        generation = SimpleNamespace(_streaming_state=state,
                                      apply_pending_streaming_sum_condition=lambda mask: None)
        with self.assertRaisesRegex(RuntimeError, "did not reach"):
            apply_reference_frame(generation)


if __name__ == "__main__":
    unittest.main()
