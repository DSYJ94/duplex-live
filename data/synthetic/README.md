# Synthetic interaction-dynamics factory (Issue #3)

Layer-3 data from docs/02-data.md: fully synthetic, fully controllable duplex
scenarios, rendered onto the 80 ms micro-turn grid. ADR-0002 rule: open models
only — no closed-API outputs anywhere in this pipeline.

## Pieces

- `timeline.py` — 80 ms grid, Event/Timeline, actualized timings, overlap stats.
- `scenarios.py` — one generator per scenario class: `hold_floor`,
  `addressee`, `backchannel`, `barge_in` (+retraction), `timed_task`,
  `delegation`, `silence_semantics`, `simultaneous`. Emits Timeline JSON.
  Same-channel starts are scheduled with `_est_ms` (measured TTS rate:
  800 ms lead-in + 80 ms/char); the filter's collision check enforces it.
- `render.py` — MiniCPM-o 4.5 zero-shot voice-cloned TTS per utterance,
  mixed at 24 kHz onto the grid; writes mixed.wav + stems + manifest.jsonl.
  Barge-in retraction enforced at mix time (v0.1). Voice pool: anchors
  (3 proven MiniCPM-o asset clips) + `--voices` json (LibriSpeech pool),
  one distinct voice per channel, deterministic per scenario_id.
- `voices.py` — stages the LibriSpeech test-clean (CC-BY-4.0) voice pool:
  one 6–10 s 16 kHz clip per speaker → `$DL/models/voices/voices.json`.
  Run on the login node inside the container.
- `filter.py` — quality gate: rule checks (clipping, same-channel event
  collisions, barge-in cut placement) + open-ASR judge pass
  (whisper-large-v3-turbo, per-utterance round-trip WER, truncation-aware).
  Writes `filter_report.jsonl` + `train_manifest.jsonl` (passing rows only).

## Run (Isambard)

```bash
sbatch synth-render.slurm            # timelines + render -> runs/synth/audio
sbatch synth-filter.slurm            # quality gate   -> filter_report.jsonl
# staging the voice pool (once):
apptainer exec --bind /projects ... $SIF python -m data.synthetic.voices --dl $DL
# outputs: runs/synth/<scenario_id>/{mixed.wav,stem_*.wav,timeline.actual.json}
#          + manifest.jsonl + filter_report.jsonl + train_manifest.jsonl
```

Quick local check (no GPU): `python -m data.synthetic.scenarios --out /tmp/tl --per-class 1`

## Known limitations (v0.2)

- Content banks are hand-written in-repo (15 explain passages, 10 stories,
  12 hard questions, 8 timed topics, 9 barge pairs). Unique at per_class≈8;
  hours-scale needs parameterized composition on top.
- Utterance durations are realized by the TTS engine; intended starts are
  honored but speech rate is what it is — `_est_ms` margins keep same-channel
  events clear, cross-channel overlap is intended.
- LibriSpeech pool voices are unvalidated until the filter's per-utterance
  WER covers a pool-rendered batch; `--anchor-prob` (default 0.5) keeps the
  proven clones in the mix. Filter WER by voice key will tell us which pool
  voices clone badly.
- Mixed-audio intelligibility is not judged (stem-level checks only) —
  overlap is partly the point; revisit if training shows issues.
