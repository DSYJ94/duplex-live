# PersonaPlex 7B duplex baseline (2026-09-16)

Runbook 02 bake-off, part 1. Method: `moshi.offline` replay of the **identical
scripted user timelines** used for MiniCPM-o (`runs/bakeoff/t*/user_in.wav`),
voice NATF2, seed 42424242, default assistant text-prompt ("wise and friendly
teacher"). One GH200, `pylibs-pplex` overlay. Output analysis from the
frame-aligned text-token stream (12.5 fps → token index × 80 ms = time).

Raw artifacts on Isambard: `/projects/u6si/duplex-live/runs/bakeoff-pplex/`
(model_out.wav + output.json per test).

## Results vs MiniCPM-o 4.5 (same inputs)

| Test | PersonaPlex 7B | MiniCPM-o 4.5 |
|---|---|---|
| T1 turn-taking | **PASS ~2–3.5 s gap** steady state; Q1/Q2 delayed ~15 s by cold-start backlog after its greeting | PASS ~2–3 s; Q1/Q2 contaminated by unprompted Chinese greeting |
| T2 barge-in | **PASS** — yields at interruption *onset* (story stops ≤0.5 s after user starts), correct direct answer ~2 s later | PASS — yields ≤1 unit (1 s) |
| T3 backchannel | **PARTIAL** — tolerates "Mm-hmm", but "Right." at 16 s derailed the topic ("Yeah, right now the biggest challenge is the climate crisis…") and the story never resumed | **PASS** — zero false yields, story unaffected |
| T4 thinking pause | **PASS** — silent through the 3 s mid-sentence pause, full correct answer after part 2 | FAIL — takes over mid-pause |
| T5 background speech | **FAIL** — engages with overheard chatter ("Yeah I saw… Right, it was crazy…") | FAIL — same failure |

## Cross-cutting

- PersonaPlex's frame grid is 80 ms (12.5 Hz text tokens) vs MiniCPM-o's
  1000 ms units — yet steady-state turn gap is similar (~2–3 s). The finer
  grid buys barge-in sharpness (sub-second yield), not lower turn latency.
- Proactive greeting at session start in both models; PersonaPlex's is
  in-character (English assistant persona), MiniCPM-o's drifts to Chinese.
- Both fail addressee detection (T5): overheard speech is treated as
  addressed. Neither model ships that policy — clear factory target (#3).
- `moshi.offline` processes ~6× faster than realtime on one GH200
  (5 timelines, ~5 min audio total, < 4 min wall including load).

## License record (Issue #4 requirement)

- Code: MIT (`personaplex/moshi`, Kyutai Moshi fork + NVIDIA changes)
- Weights: **NVIDIA Open Model License** (accepted via HF gate by repo owner's
  account; token at `$DL/.hf_token`)
- Moshi base: CC-BY-4.0
- Program rule: PersonaPlex outputs are **evaluation baselines only** — never
  training data (same rule as closed-API brain outputs; ADR discipline).
