# evals/

Evaluation harness (see `docs/05-evaluation.md`). Runs on every checkpoint.

- `interactivity/` — FD-bench-style turn-taking latency, interruption/backchannel scenarios
- `dimensions/` — mini TimeSpeak/CueSpeak-style timing & proactivity tests
- `intelligence/` — no-regression suite vs base Qwen3-Omni (Audio MultiChallenge, IFEval)
- `contamination/` — train/eval overlap checks

Gate thresholds: see docs/07 Gate B.
