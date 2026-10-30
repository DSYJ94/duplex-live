# 00 — Pretraining Decision: From Scratch vs. Open Source

> **30 October 2026 review:** This is a historical exploratory design, not the current approved execution plan. Its cost/time estimates and proprietary-model implementation claims are not verified planning inputs. The absolute recommendation to train audio/video pathways from scratch is superseded: a qualified released native duplex model can reuse pretrained language and audio components. Foundation pretraining is not currently recommended. See the [current evidence-led foundation and cost review](NATIVE-DUPLEX-FOUNDATION-COST-REVIEW-2026-10-30.md) and its bounded base/runtime comparison before further investment. The original proposal below remains intact as history.

## The question

Duplex Live is a proprietary model. Do we (a) pretrain the entire multimodal duplex model from scratch, or (b) adapt an open-source base and train the duplex/live capability on top?

## What the references tell us

- **Thinking Machines trained from scratch** — and their post explains *why it mattered*: encoder-free early fusion (dMel → lightweight embedding, 40×40 patch hMLP for video, flow head for audio out) co-trained with the transformer means the interaction dynamics are learned end-to-end. No Whisper-style encoder whose frozen representations bottleneck interactivity.
- **OpenAI's GPT-Live** is a systems achievement layered on frontier models: stateful streaming inference, asynchronous delegation to GPT-5.5-class models, context compaction, transport optimization. The *model* is full-duplex; much of the magic is serving.
- **Prior open art** (Moshi, PersonaPlex, Nemotron VoiceChat) proves full-duplex training works at small scale on top of pretrained backbones — but those models sacrifice intelligence for latency.

**Conclusion:** interactivity must be in the model (non-negotiable). Whether the *text/intelligence core* must be pretrained by us is a separate, economic question.

## Options compared

| | A. Full scratch | B. Open-source base + duplex adaptation | C. Hybrid staged (recommended) |
|---|---|---|---|
| Time to first duplex demo | 9–15 months | 3–5 months | 3–5 months (A-track), scratch follows |
| Pretraining compute | $15M–$60M+ (70B–300B MoE class) | $1.5M–$5M (continued pretrain + duplex SFT) | Phase A cheap; Phase B funded by evidence |
| Data risk | Must solve text *and* duplex data at once | Reuses base model's intelligence | Duplex recipe validated before scratch spend |
| Proprietary control | Total | Base weights are open (license-dependent) | Phase B yields fully proprietary model |
| Failure mode | A year in, discover duplex data recipe is wrong | Base model ceiling on intelligence | Coordination overhead of two tracks |

## Recommended plan: staged hybrid with explicit gates

### Phase A — Duplex adaptation on an open-source base (Quarters 1–2)

- **Base selection criteria:** permissive license (Apache-2.0 / MIT), strong text intelligence, MoE preferred for latency (low active params), multimodal-ready tokenizer. Candidate class: Qwen-Omni-family-style open models, or a text-only MoE + our own A/V pathways.
- Train the duplex layer: micro-turn tokenization, simultaneous input/output streams, turn-taking behavior, backchanneling, interruption handling, time-awareness tokens.
- Stand up the serving stack (stateful streaming sessions) and the eval harness (FD-bench-style) **in parallel** — these are on the critical path regardless of which model wins.

### Gate 1 (end of Q2): Does Phase A hit interactivity targets?

Targets: turn-taking latency < 600 ms, FD-bench-style average ≥ 60, no VAD in path, stable 10-min sessions. If yes → ship a v0 product and fund Phase B. If no → fix data/training, do **not** start scratch pretraining with an unproven recipe.

### Phase B — From-scratch proprietary pretraining (Quarters 3–8)

- Encoder-free early-fusion architecture (see `01-architecture.md`), MoE, target ~12B active params as the first scratch scale point (TML's `Interaction-Small` reference: 276B total / 12B active — we should start smaller, e.g. 30–80B total / 3–8B active, and scale after serving is proven).
- Reuse Phase A's validated data pipeline, evals, and serving; swap the intelligence core for our own pretrained weights.
- The duplex/live adaptation recipe transfers directly — that is the whole point of the hybrid.

### Gate 2 (end of Q4): Scratch data mixture validated at 1–3B scale before committing full pretraining run.

## Decision summary

| Decision | Choice | Rationale |
|---|---|---|
| Duplex/live in model vs. harness | **In model (must)** | VAD/turn-detector harnesses cap interactivity; both references removed them |
| Text core, short term | **Open-source base** | Speed, cost, risk isolation |
| Text core, long term | **Pretrained from scratch (proprietary)** | Full control, co-trained early fusion, no license ceiling |
| Audio/video pathways | **From scratch, always** | dMel/flow-head/patch-MLP co-training is the source of interactivity |
| Delegation model | **Existing frontier model first, own model later** | Delegation is async; correctness matters more than ownership at v0 |
