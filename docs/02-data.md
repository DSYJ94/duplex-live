# 02 — Data Strategy

The data problem is the program's real moat. Duplex interaction data — time-aligned, overlapping, multi-modal — barely exists in the wild at scale.

## Data layers

### Layer 1: Intelligence foundation (mostly solved by base choice)
- Phase A: inherited from open-source base; we add domain SFT only.
- Phase B: standard text/code/math pretraining mixture (licensed + synthetic), sized per scaling-law runs at 1–3B before the main run.

### Layer 2: Duplex conversation corpus (the hard part)

Target: tens of thousands of hours of **time-aligned two-channel (or multi-channel) conversation**, plus aligned video where available.

| Source | Volume potential | Duplex quality | Notes |
|---|---|---|---|
| Recorded human conversations (call-center style, licensed) | High | High | Natural overlap, backchannels, interruptions; consent/licensing work required |
| Podcasts / interviews / talk shows (licensed) | Medium | Medium-High | Good turn dynamics; clean audio needed for micro-turn alignment |
| Meetings / panel discussions | Medium | High | Multi-speaker overlap; harder attribution |
| Synthetic duplex via TTS + scripted interaction dynamics | Very high | Tunable | The scaling lever — see below |
| Human-in-the-loop sessions with our own v0 model | Growing | Perfectly on-distribution | Flywheel starts after Phase A ships |

**Alignment requirement:** every stream must be micro-turn aligned (200 ms grid) with precise timestamps, speaker attribution, and overlap preserved — not segmented into turns. Build the alignment toolchain (forced alignment, diarization with overlap, QC) in Quarter 1; it gates everything.

### Layer 3: Synthetic interaction dynamics

Real conversations give texture; synthetic data gives coverage of rare-but-critical behaviors:

- **Interruption taxonomy:** barge-in to correct, barge-in to change topic, barge-in the model should ignore, collaborative overlap, backchannels.
- **Silence semantics:** thinking pause vs. finished thought vs. inviting the model to speak — generated with controllable prosody.
- **Simultaneous speech:** live translation, sports commentary, pronunciation correction overlays.
- **Time-aware tasks:** timers, pacing, "every N seconds" instructions (mirrors TML's TimeSpeak/CueSpeak evals — train distribution must cover the capability, not the test).
- **Delegation scenarios:** conversations where the right move is to consult the background model, with streamed result re-integration examples.
- **Refusal/over-refusal speech data** via TTS over text refusal distributions (see `06-safety.md`).

Pipeline: script interaction scenarios → multi-voice TTS with naturalistic prosody/overlap → render onto the 200 ms micro-turn grid → filter with model-based quality judges.

### Layer 4: Video-grounded streams

- Action-centric video with audio instructions (rep counting, temporal start/stop callouts — Charades-style but as *training* streams).
- Screen-sharing / co-browsing style video+speech for the collaboration use case.
- Proactive-visual-moment data: questions whose answers appear at specific video timestamps.

## Mixture plan (Phase A duplex adaptation)

| Component | Share (tokens) |
|---|---|
| Duplex real conversations | 35% |
| Synthetic interaction dynamics | 25% |
| Text intelligence replay (base-model distribution) | 25% |
| Video-grounded streams | 10% |
| Safety/refusal speech | 5% |

Anti-forgetting: always replay base-distribution text; audit intelligence benchmarks at every checkpoint.

## Governance

- Provenance tracking per stream; licensed vs. synthetic tagging in metadata.
- PII scrubbing on all recorded audio; voice-consent registry for any cloned/contributed voices.
- Dedup against eval sets (contamination checks on FD-bench-style and audio intelligence benchmarks).
