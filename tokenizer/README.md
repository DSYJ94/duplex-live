# tokenizer/

Canonical micro-turn tokenizer/streamer — the single implementation shared
by training and serving. **Format spec: SPEC.md** (start there).

- Chunks all streams onto the **80 ms grid** (12.5 Hz — Qwen3-Omni's
  native AuT/Talker rate; the original 200 ms plan was superseded by
  docs/09).
- Produces the interleaved `in_t out_t` causal sequence of docs/09 §1:
  in-stream activity (user + background mix), Thinker emissions (words,
  `<sil>`, `<yield>`, `<delegate>`, `<t+Ns>`), Talker voicing
  continuation, `<barge-in>` markers.
- Reserved vocab in `vocab.py`; `register_into()` adds them to the HF
  tokenizer before embedding resize.
- One codebase, two consumers: training sampler and streaming-session
  server. Divergence here = trainer-sampler misalignment.

```bash
python -m tokenizer.test_roundtrip --batch runs/synth-big2/audio
```

Validated over all local batches (860 scenarios at authoring).
