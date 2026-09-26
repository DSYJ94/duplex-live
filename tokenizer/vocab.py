"""Reserved vocabulary for the 80 ms micro-turn stream (Issue #2, docs/09 §2c).

New tokens added to Qwen3-Omni's 151,643-token vocab for duplex control.
IDs are assigned at training time via `register_into()` (resize embeddings,
init new rows from mean-of-neighbors); this module is the single registry
shared by the factory serializer, the training sampler, and the serving
tick loop.
"""

from __future__ import annotations

SIL = "<sil>"
SPEAK = "<speak>"          # keep voicing: inside an utterance, no new word
YIELD = "<yield>"          # stop speaking, listen (after barge-in)
DELEGATE = "<delegate>"    # fire async RPC at the boundary
BARGE_IN = "<barge-in>"    # in-stream marker: other channel barged in

# Explicit timestamps (docs/09 §2d): emitted at the start of an assistant
# event that follows a >= 1 s gap, bucketed. Buckets are seconds.
TIMING_BUCKETS_S = (1, 2, 5, 10, 30, 60)

# v2 label contract (2026-09-26, ADR-0004): every tick inside an
# assistant utterance carries WORD (a new word starts) or SPEAK (the
# current voicing continues) — the v1 contract labeled continuations
# <sil>, structurally making the grid ~88% silence and starving the
# word pole (EXPERIMENTS.md probes 5-6: loss-side dosing refuted).
ALL_SPECIAL = [SIL, SPEAK, YIELD, DELEGATE, BARGE_IN] + \
    [f"<t+{n}s>" for n in TIMING_BUCKETS_S]


def timing_token(gap_ms: int) -> str | None:
    """Bucketed <t+Ns> token for a silence gap; None if below threshold."""
    gap_s = gap_ms / 1000.0
    if gap_ms < 1000:
        return None
    for n in TIMING_BUCKETS_S:
        if gap_s <= n:
            return f"<t+{n}s>"
    return f"<t+{TIMING_BUCKETS_S[-1]}s>"


def register_into(tokenizer) -> list[int]:
    """Add the reserved tokens to a HF tokenizer; returns new token ids.
    Call before model.resize_token_embeddings()."""
    return tokenizer.add_tokens(ALL_SPECIAL, special_tokens=True)
