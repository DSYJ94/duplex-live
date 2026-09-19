# Micro-turn stream format — the 80 ms interleaved causal sequence

Issue #2 / docs/09 §1. This is the contract between the synthetic factory
(`data/synthetic/`), the training sampler (Issue #7), and the serving tick
loop. Canonical code: `grid.py` (build), `serialize.py` (format),
`vocab.py` (reserved tokens), `test_roundtrip.py` (invariants).

## Grid

- One tick = **80 ms** (12.5 Hz) = one AuT audio frame = one Talker codec
  frame. No resampling anywhere (Qwen3-Omni native rate both sides).
- A scenario serializes to `n_ticks = ceil(duration_ms / 80) + 1` ticks.

## Per-tick record

| field | meaning |
|---|---|
| `t` | tick index (the audio-IN slot is implicit: AuT frame `t` attaches here; input audio = **stem_user + stem_background mix** — background rides the in-slot so the model learns audible-not-addressed, T5) |
| `in` | input-stream activity this tick (user OR background speech) |
| `out` | the Thinker's emission `x_t`: a word, `<sil>`, `<yield>`, `<delegate>`, or `<t+Ns>` |
| `voi` | assistant speech span active — Talker continues voicing from prior text (ticks inside an assistant event's actual span) |
| `mark` | in-stream markers, currently `<barge-in>` at the interruption tick |

Emission precedence: word > `<yield>`/`<delegate>` > `<t+Ns>` > `<sil>`.
Words shift forward to the next free tick rather than collide-drop.

## Semantics encoded

- **`<sil>`**: no new text; Talker silent outside `voi` spans, continuing
  inside them.
- **`<yield>`**: at the tick where a barge-in CUT an assistant utterance
  (== the barge instant). Post-cut words are NOT emitted — retraction is
  data (the truncated prefix is spoken, the rest vanishes).
- **`<delegate>`**: at marker events (`kind=marker`, delegate boundary).
- **`<t+Ns>`**: at the start of an assistant event following a ≥1 s gap;
  buckets {1,2,5,10,30,60} s.
- **`<barge-in>`**: marker on the in-stream at the barge tick.

## File formats

Machine: JSONL — one header line (scenario id/class/n_ticks/audio refs/
meta), then one line per non-default tick. Human: `serialize.to_compact`
renders one char per tick (`.` `u` `A` `W` `Y` `D` `T` `|`).

## v0 limitations (documented upgrades)

- Word→tick alignment is char-proportional within each utterance's actual
  span (truncation-aware: post-cut words dropped, 320 ms/word estimate).
  Upgrade path: forced word alignment (whisper-timestamped / MFA) writing
  per-word times into `timeline.actual.json` — format unchanged.
- Words are atomic tokens here; real BPE tokenization happens in the
  training sampler (multi-token words span ticks with `voi` continuation).
- Audio slots are references (stems); AuT/codec attach at training time.

## Round-trip guarantee

`serialize.from_jsonl(serialize.to_jsonl(g)) == g`, plus invariants
tested over every local batch (860 scenarios at authoring time): word
counts (full non-truncated + strict truncated prefixes), yields == cuts,
barge markers == barge events, delegates == markers, overlap ticks ==
timeline cross-channel overlap.
