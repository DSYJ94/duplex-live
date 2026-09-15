# MiniCPM-o 4.5 duplex baseline (2026-09-15)

Runbook 02 bake-off, part 2. Harness: `evals/bakeoff/minicpm_bakeoff.py`
(scripted 1 s-chunk user timeline; user prompts synthesised with the model's
own TTS, voice `system_ref_audio_2`; assistant voice `HT_ref_audio`).
Raw artifacts (user_in.wav / model_out.wav / log.jsonl per test) on Isambard:
`/projects/u6si/duplex-live/runs/bakeoff/` — not committed (audio stays on
Lustre; becomes training-data seed per docs/02).

Setup: one GH200, transformers 4.51.0 overlay, offline (faster-than-realtime)
execution; all timings in 1 s chunks of the user timeline.

## Results

| Test | Outcome | Evidence |
|---|---|---|
| T1 baseline turn-taking | **PASS, ~2–3 s gap** | steady-state user-end→model-speech-start 1–4 chunks (p50 ≈ 2.5 s) across 10 Qs; all answers correct (ASR artifact: "Romeo" → "Rau mial") |
| T2 barge-in | **PASS** | interruption mid-sentence at chunk 16 → turn ended chunk 17 (≤1 unit to yield), correct short answer to the new question at 21 |
| T3 backchannel | **PASS** | "mm-hmm"/"right"/"uh-huh" at chunks 10/16/22 during continuous 7–44 answer — zero yields, zero false turn-ends |
| T4 thinking pause | **FAIL** | 3 s mid-sentence pause → model took over (chunks 5–8) before the sentence finished; correct answer after part 2 |
| T5 background speech | **FAIL** | model engaged with overheard chatter at length (chunks 6–20); no addressee detection |
| T6 long session | not yet run | harness supports it (`--tests t6`, ~900 chunks) |

## Cross-cutting observations

- **Unprompted proactive greeting** at session start in every run (chunks 0–6),
  usually in Chinese ("嗯，如果说想去看美丽的风景啊…") despite an English system
  prompt — likely primed by the assistant reference voice / demo bias.
  Contaminated T1's first two questions; steady-state numbers above exclude it.
- **Throughput**: wall per 1 s chunk p50 0.73 s, max 1.77 s (includes LLM +
  TTS + vocoder) → faster than realtime on one GH200 with headroom.
- **Telemetry**: the duplex API returns per-chunk `cost_llm / cost_tts /
  cost_token2wav` — reuse this shape for our own clock layer (Issue #6).

## Implications for Layer-A (docs/09)

T4/T5 are pure *policy* failures — exactly the classes our synthetic
interaction-dynamics factory (Issue #3) must generate: mid-sentence pauses
with hold-floor examples, and background/addressee discrimination.
MiniCPM-o's 1 Hz unit caps response gap at ~1–2 s by construction; the 80 ms
tick target exists to beat this. See `docs/notes-minicpm-layera.md`.
