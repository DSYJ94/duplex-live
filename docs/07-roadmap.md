# 07 — Roadmap (Mini Open-Source Program, Isambard-AI)

Supersedes the lab-scale roadmap (see git history / ADR-0001). Scale: 1–3
people, free GH200 compute, ~$0–2k cash, 2–3 quarters to a working demo.

## Q1 — Foundations + fast demo

- [ ] **W1: aarch64 environment bring-up** on Isambard (PyTorch, transformers,
      audio stack, vLLM/SGLang ARM builds) — blocks everything
- [ ] Micro-turn tokenizer/streamer (200 ms grid) — `tokenizer/`
- [ ] Synthetic interaction factory v0: scripted interruption/silence/timing
      scenarios → open TTS → grid-aligned → judge filter — `data/synthetic/`
- [ ] **PersonaPlex quick demo** (duplex works out of box) to validate serving
      loop and give the interactivity reference
- [ ] Eval harness v0: latency sim + FD-bench-style scenarios — `evals/`
- [ ] Delegation boundary: `<delegate>` → async RPC → Kimi API → streamed
      re-integration (works with an unmodified base model first)

### Gate A (end Q1): PersonaPlex-based duplex demo with live delegation working end-to-end.

## Q2 — The Duplex Live model

- [ ] Qwen3-Omni duplex fine-tune per `docs/08-duplex-finetune-recipe.md`
      (Phase 1 frozen-backbone, then Phase 2 full FT + replay)
- [ ] Timing tokens + silence-loss training; anti-collapse monitoring
- [ ] Internal timing/proactivity evals (TimeSpeak/CueSpeak-style, mini versions)
- [ ] Single-node serving demo: streaming session + WebRTC on one GH200
- [ ] Head-to-head: our fine-tune vs PersonaPlex on interactivity; vs base
      Qwen3-Omni on intelligence (no-regression gate)

### Gate B (end Q2): fine-tuned Qwen3-Omni matches PersonaPlex interactivity with clearly higher intelligence, OR a documented interactivity ceiling justifies Track 2 scratch.

## Q3 (conditional) — Scratch track

- [ ] 1B pilot (native micro-turn format, encoder-free audio path) — days of compute
- [ ] 3B main run only with debugged data pipeline + evals (~2–4 wks on 32–64 GH200s)
- [ ] Matched-scale comparison vs fine-tune track → write up

## Team / budget

| Item | Scale |
|---|---|
| People | 1–3 |
| Compute | Isambard-AI allocation (free); biggest single job ~80 GH200-days |
| Cash | ~$0–2k (TTS APIs if used, data misc) |

## First actions (this week)

1. Confirm Isambard allocation details (GPU-hours, partition, container policy).
2. aarch64 bring-up spike.
3. PersonaPlex weights download + license review (NVIDIA OML).
4. Tokenizer/streamer scaffolding in `tokenizer/`.
