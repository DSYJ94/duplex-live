# 03 — Training Program

## Stage 0 — Infrastructure & toolchain (Q1, parallel with everything)

- Micro-turn tokenizer/streamer (200 ms grid) shared by training and serving — one canonical implementation.
- Streaming-session training sampler: presents interleaved in/out micro-turn sequences exactly as inference will see them (trainer–sampler alignment below).
- Eval harness wired into CI of checkpoints from day one (`05-evaluation.md`).

## Stage 1 — Phase A: duplex adaptation (Q1–Q2)

1. **Continued pretraining (duplex CPT):** base LLM + new audio/video pathways trained on the Layer-2/3 mixture with next-micro-turn prediction over the interleaved stream. Objectives: audio token/feature prediction, text prediction, *silence/timing prediction* (the model must learn when NOT to speak).
2. **Duplex SFT:** behavior-level tuning — interruption etiquette, backchannel policy, proactive triggers, delegation decisions.
3. **Preference optimization on interaction quality:** pairwise human/model judgments on *timing and appropriateness*, not just content. Reward model must score latency-aware behavior (right thing, right moment).

Compute envelope: single-digit $M; 100–500 H100/B200-class GPU-months depending on base size.

## Stage 2 — Phase B: scratch pretraining (Q3–Q8, gated)

1. **Scaling-law runs at 1–3B** to fix the multimodal mixture and hyperparameters (Gate 2).
2. **Main run:** MoE, 30–80B total / 3–8B active, co-trained dMel-embedding + patch-hMLP + flow head. Bitwise trainer–sampler alignment from the start (below).
3. **Post-training:** same duplex SFT + preference stack as Stage 1, plus safety stage (`06-safety.md`).

## Trainer–sampler alignment (adopting TML's hard-won lesson)

Bitwise alignment between training and serving numerics is a stability and debugging superpower for duplex models:

- **Batch-invariant kernels** with <5% e2e overhead budget; acceptable even at parity.
- **Deterministic comms:** NVLS-based all-reduce / reduce-scatter, deterministic on target GPU generation; alignment across differing parallelism layouts (SP vs. TP).
- **Attention:** consistent accumulation order between prefill and decode (left-aligned fixed-size split-KV, e.g. 4096-token splits).

## Checkpoint policy

- Every checkpoint auto-scored on: turn-taking latency (simulated), FD-bench-style interactivity, audio intelligence suite, text intelligence regression suite, safety refusal battery.
- Promotion requires: no regression >2% on text intelligence, monotonic interactivity improvement, safety pass.

## Delegation training

- Stage the background model as a fixed, strong reasoner first; train the core's *delegation policy* (when to delegate, how to stall, how to weave results) with the background model frozen.
- Later: co-adapt via preference data on end-to-end task outcomes.

## Key training risks

| Risk | Mitigation |
|---|---|
| Model collapses to turn-taking (easy local optimum) | Heavy overlap/silence data share; timing-prediction loss term; eval gating on simultaneous-speech tasks |
| Intelligence regression from duplex adaptation | Text replay, per-checkpoint regression gates, MoE capacity headroom |
| Prosody/latency uncorrelated with content quality | Timing-aware reward model; FD-bench-style metrics in the loop |
| Flow-head audio quality ceiling | Benchmark against neural-codec alternatives early (Gate review in Q2) |
