# TypeSafe Jev / System One — research notes (2026-09-20)

Sources: docs.typesafe.ai (introduction, how-to-build-with-system-one,
model-jaggedness/jev-1.13, primitives: noul/score/choice, confidence,
sdk/python/usage) + the typesafe-ai skill. Companion to docs/05 (eval)
and docs/02 (data).

## What it is (plain words)

Jev is not a chatbot. It is a **judgment engine**: you hand it some
context ("state") and a set of pre-structured questions, and it returns
numbers — probabilities — not prose. ~100 ms per batch, many questions
answered in parallel. Three question shapes ("primitives"):

- **Noul** — a yes/no judgment as a probability (0..1). No separate
  confidence; the number is the whole story. One question per condition.
- **Choice** — picks one option from a list you define, with a
  probability spread and a confidence number.
- **Score** — grades on ordered descriptive levels ("broken but
  workaround exists" vs "blocking"), returns a probability-weighted
  position you can rank and average.

Methodology ("System One"): code owns control flow, math, and
composition; the model only supplies narrow, typed common sense. Ask
many small questions over shared state in one call; combine answers with
weights/thresholds in code.

## Jaggedness (known weaknesses) — mapped to us

Jev is literal-minded and **bad at numbers, dates, and durations** —
exactly the things our timing metrics live on. Consequences for us:
**all latency/silence/timing judgments stay in code** (they already
are); Jev only judges *meaning*. Also: multi-hop reasoning is weak
(keep questions single-hop), distractor-heavy state degrades it (send
minimal state), and it cannot generate text (never use it for content).

## Where Jev fits duplex-live (ranked)

1. **Eval harness content scoring (Issue #6)** — the semantic half our
   energy metrics lack. Per recorded session, one request with parallel
   Nouls/Scores over the transcript: "does the reply answer the question
   that was asked?", "does the model's response engage with the
   background conversation it should have ignored?" (T5 semantic
   confirmation), "is the reply after interruption a genuine answer to
   the interruption?" (T2), Score on answer quality. Thresholds route
   verdicts: auto/pass, review, fail.
2. **Factory bank audit (Issue #3 tail)** — structural validators catch
   shape; Jev catches meaning: Noul "does this yield actually answer
   this barge-in?" per pair; Score naturalness per bank item. One-time
   per item at expansion — cheap.
3. **Phase-1 sampler audit** — Score scenarios on naturalness /
   class-typicality once; code turns scores into mixture weights and
   review queues (composite-scoring pattern).
4. **Layer-2 real-corpus VALIDATION** — Jev labels on CANDOR/PMQs
   segments as silver-standard to *validate* open-model labelers (see
   ADR line below).

## The ADR-0002 line (important)

Jev is a closed API. Same rule as Kimi: its outputs never enter training
data. Sharpened for this:

- **Eval/audit side (scores, verdicts, reports): YES** — scores gate and
  describe; they are metadata, not training content.
- **Training supervision (labels our model learns from, e.g. silence-
  semantics labels on real corpora): NO** — that would be closed-API
  outputs becoming training targets. Use open models (Qwen) for
  train-side labeling; Jev may audit their accuracy (eval-side).

## Where Jev does NOT fit

- The 80 ms tick loop (latency + network; the policy must live in the
  model's own weights after Layer-A training — that's the whole point).
- TTS/ASR (specialist open models already serve us).
- The delegation brain (that's generation — Kimi/ACP).

## Integration shape

Python SDK `typesafe_sdk`; `TYPESAFE_API_KEY` env var; sync/async
clients; `client.system_one(state, {qid: Noul/Choice/Score...})`.
Retries + `TypeSafeAPIError` handling. Key handling follows the
`.hf_token` pattern on the cluster (600-perm file, never committed).
