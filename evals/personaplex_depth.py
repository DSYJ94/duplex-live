"""Match the pinned native depth decoder's streaming attention during training.

The vendor RingKVCache labels its oldest slot as a future position when full
(delta <= 0). The streaming attention mask therefore excludes that slot. The
unstreamed native helper otherwise includes it. The pinned depth context is
None, so its ring capacity is the 16 depth steps. Only the batched training
call uses context 15; streaming inference and vendor source remain untouched.
Native per-codebook logit parity must establish this compatibility explicitly.
"""
from contextlib import contextmanager


@contextmanager
def native_depth_context(model):
    changed = []
    for layer in model.depformer.layers:
        attention = layer.self_attn
        if attention.is_streaming or attention.context is not None or attention.weights_per_step != 16:
            raise RuntimeError("Unexpected pinned PersonaPlex depth context/state")
        changed.append(attention)
        attention.context = 15
    try:
        yield
    finally:
        for attention in changed:
            attention.context = None


def native_audio_logits(model, targets, hidden):
    with native_depth_context(model):
        return model.forward_depformer_training(targets, hidden)
