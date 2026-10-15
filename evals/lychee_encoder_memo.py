"""Bounded inference-only encoder memo for the pinned native Lychee HF loop.

The loop passes the same immutable waveform-feature tensors at each generated
token. Cache only within that exact tensor pair; a new arrived chunk invalidates
the entry. Direct incremental encoder calls always execute normally.
"""
from contextlib import contextmanager


@contextmanager
def encoder_memo(model):
    import torch
    original_forward = model.forward
    original_encoder = model.encoder.forward
    state = {"active": None, "key": None, "value": None}
    stats = {"native_encoder_calls": 0, "cache_hits": 0, "direct_encoder_calls": 0}

    def encode(*args, **kwargs):
        active = state["active"]
        if active is not None and state["key"] is not None and all(a is b for a, b in zip(active, state["key"])):
            stats["cache_hits"] += 1
            return state["value"]
        result = original_encoder(*args, **kwargs)
        if active is None:
            stats["direct_encoder_calls"] += 1
        else:
            stats["native_encoder_calls"] += 1
            # Keep strong references so allocator pointer reuse cannot hit.
            state["key"], state["value"] = active, result
        return result

    def forward(*args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError("Encoder memo is inference-only")
        wavs, lens = kwargs.get("wavs"), kwargs.get("wav_lens")
        if (wavs is None) != (lens is None):
            raise RuntimeError("Incomplete native audio feature pair")
        previous = state["active"]
        state["active"] = (wavs, lens) if wavs is not None else None
        try:
            return original_forward(*args, **kwargs)
        finally:
            state["active"] = previous

    model.forward, model.encoder.forward = forward, encode
    try:
        yield stats
    finally:
        model.forward, model.encoder.forward = original_forward, original_encoder
        state.clear()
