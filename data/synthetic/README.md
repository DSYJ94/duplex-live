# Synthetic interaction-dynamics factory (Issue #3)

Layer-3 data from docs/02-data.md: fully synthetic, fully controllable duplex
scenarios, rendered onto the 80 ms micro-turn grid. ADR-0002 rule: open models
only — no closed-API outputs anywhere in this pipeline.

## Pieces

- `timeline.py` — 80 ms grid, Event/Timeline, actualized timings, overlap stats.
- `scenarios.py` — one generator per scenario class: `hold_floor`,
  `addressee`, `backchannel`, `barge_in` (+retraction), `timed_task`,
  `delegation`, `silence_semantics`, `simultaneous`. Emits Timeline JSON.
- `render.py` — MiniCPM-o 4.5 zero-shot voice-cloned TTS per utterance
  (per-channel voice: assistant/user/background), mixed at 24 kHz onto the
  grid; writes mixed.wav + stems + manifest.jsonl.

## Run (Isambard)

```bash
sbatch synth-render.slurm            # generates timelines, renders all
# outputs: /projects/u6si/duplex-live/runs/synth/<scenario_id>/{mixed.wav,stem_*.wav,timeline.actual.json}
#          + manifest.jsonl
```

Quick local check (no GPU): `python -m data.synthetic.scenarios --out /tmp/tl --per-class 1`

## v0 limitations

- Voices: 3 fixed reference clips from the MiniCPM-o snapshot assets. Widen the
  voice pool (more ref clips, varied prosody) before scaling hours.
- Utterance durations are realized by the TTS engine; intended start times are
  honored but speech rate is what it is (~65 ms/word estimates in generators
  are only for placing *subsequent* intended starts).
- Quality filter (judge pass) not yet wired; v0 filter = duration/clipping
  sanity in the manifest.
- Barge-in retraction is enforced at mix time (v0.1): any utterance overlapped
  by a `barge_in` event on another channel is truncated at the barge instant
  with a 100 ms fade, and the manifest records the cut.
