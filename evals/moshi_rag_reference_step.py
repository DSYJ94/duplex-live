"""Apply one reference row for the single active slot in the native replay."""
import torch


def apply_reference_frame(generation):
    state = generation._streaming_state
    if state is None or state.batch_size != 1:
        raise RuntimeError("Reference replay requires one active streaming slot")
    pending = state.pending_streaming_sums[0]
    expected = None if pending is None else pending[0]
    # This explicit operation belongs to the upstream batch runner, not step().
    generation.apply_pending_streaming_sum_condition(torch.ones(1, dtype=torch.bool))
    actual = state.condition_streaming_sum[0, 0]
    if expected is None:
        if torch.count_nonzero(actual):
            raise RuntimeError("Completed reference left stale conditioning")
    elif not torch.equal(actual, expected):
        raise RuntimeError("Queued reference row did not reach active conditioning")
    return float(actual.float().abs().sum())
