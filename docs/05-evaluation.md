# 05 — Evaluation

Interactivity must be measured as rigorously as intelligence. Both references converge: existing benchmarks under-measure interaction, so we adopt public suites **and** build internal ones.

## Suite 1 — Streaming interactivity

| Benchmark | Measures | v1 target |
|---|---|---|
| FD-bench-style turn-taking latency | User→model response gap | < 0.6 s p50 (TML reference: 0.40 s; GPT-realtime-minimal: 1.18 s) |
| FD-bench v1.5-style (interruption, backchannel, talking-to-others, background speech) | Interaction quality | ≥ 65 avg (TML: 77.8; best commercial: ~54) |
| Tool-use duplex response quality (FD-bench v3-style) | Quality + pass@1 with delegation | match or beat non-duplex frontier |

## Suite 2 — New interactivity dimensions (internal, to be open-sourced later)

- **TimeSpeak-style:** initiate speech at user-specified times with correct content ("remind me to breathe every 4 s").
- **CueSpeak-style:** speak at the semantically right moment, including simultaneously with the user (code-switch correction, live translation).
- **RepCount-A-style:** stream video, count repeated actions aloud, off-by-one accuracy.
- **ProactiveVideoQA-style:** stay silent until the visual moment that answers the question; PAUC@ω=0.5 (silence baseline = 25).
- **Temporal callouts (Charades-style):** say "start"/"stop" at action boundaries; temporal IoU.

Graded by LLM judge on *meaning + timing window*; both required for credit.

## Suite 3 — Intelligence (no regression allowed)

- Audio: Audio MultiChallenge, BigBench Audio, VoiceBench IFEval.
- Text: IFEval, standard knowledge/reasoning suites for the chosen base class.
- Streaming video QA (QIVD-style) for multimodal intelligence.

## Suite 4 — Systems & robustness

- Long-session soak (1h+, compaction cycles, reconnects).
- Packet-loss / jitter / clock-drift degradation curves.
- Concurrent-session capacity at frame-schedule SLO.
- Human eval: side-by-side conversation naturalness (MOS-style), task completion with human-in-the-loop.

## Eval governance

- Contamination checks against training corpus every data drop.
- Eval harness runs on every checkpoint; dashboards track interactivity↔intelligence frontier over time — the goal is to push the Pareto front, not trade one for the other.
- Internal interactivity benchmarks are versioned and rotated to avoid overfitting the training mixture to the test.
