# ADR-0002: Base Model & Scale — Mini Open-Source Program on Isambard-AI

**Status:** Accepted (2026-09-14) — plan of record

## Context

The program has free compute on Isambard-AI (5,448 GH200 superchips, aarch64,
96 GB HBM each) and no commercial-lab budget. Goal: a full-duplex, time-aware
interaction model with delegation to a smarter background model, as a mini
open-source version of GPT-Live / TML interaction models.

## Decision

**Sequential tracks, not either/or:**

### Track 1 — Fine-tune (primary, weeks 1–8)
- **Base: Qwen3-Omni 30B-A3B** (Apache-2.0). Highest intelligence of any open
  omni model; ~3B active params/token leaves GH200 headroom for duplex machinery.
  Duplex behavior is fine-tuned in per `docs/08-duplex-finetune-recipe.md`.
- **Reference: PersonaPlex 7B** (NVIDIA, Jan 2026; MIT code / NVIDIA OML weights
  on CC-BY-4.0 Moshi base). Best open full-duplex behavior — the interactivity
  bar our fine-tune must match, and a quick demo path. Research reference only
  until NVIDIA license terms are reviewed.
- **Fallback: Qwen2.5-Omni 7B** if Qwen3-Omni aarch64 porting is painful.

### Delegation (from day one)
- Async RPC boundary; background model = Kimi API for the demo, swappable to a
  self-hosted open 32B+ model later.
- Legal note: base-model licenses (Apache-2.0/MIT/CC-BY) do not restrict which
  external services the system calls at inference. Bright line: closed-API
  outputs may flow through the live path but **never into the training corpus**
  (provider ToS). Synthetic training data comes from self-hosted open models.

### Track 2 — From-scratch (gated, months 3–6)
- 1B pilot → 3B main run, native micro-turn format, encoder-free audio pathway.
- Compute: ~30–80 GH200-days for 3B at 60–150B tokens → 32 GPUs for 1–3 days
  (pilot) to ~2–4 weeks (over-trained 3B). Comfortably inside Isambard.
- **Gate:** only proceeds if Track 1 demonstrates an interactivity ceiling
  (timing behavior won't hold / collapses to turn-taking) with debugged data
  pipeline and evals. The scratch-vs-fine-tune comparison at matched scale is
  the research contribution.

## Hardware constraint

Isambard-AI is **aarch64-only**. All training/serving containers must be
ARM-native from day one; verify PyTorch/transformers/vLLM aarch64 wheels in
week 1. Budget one sprint for environment porting.

## Consequences

- Cash budget ~$0–2k (data/TTS/misc); compute is free.
- Timeline: working duplex+delegation demo in 2–3 quarters with 1–3 people.
- Proprietary/commercial use is served fine by Apache-2.0 bases; scratch is
  motivated by capability ceiling and research novelty, not licensing.
