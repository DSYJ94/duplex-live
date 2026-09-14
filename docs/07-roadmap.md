# 07 — Roadmap (Mini Open-Source Program, Isambard-AI)

Supersedes the lab-scale roadmap (ADR-0001). Scale: 1–3 people, Isambard-AI
compute, ~$0–2k cash.

## ⚠️ The binding constraint: allocation terminates 2026-10-21

Confirmed 2026-09-14 (BriCS portal): **4,974 node-hours** on Isambard-AI
Phase 2 (~19,900 GH200-hours), **~5 weeks remaining**. The hours are ample;
the deadline is not. Everything below is reordered around it.

## Week 1–2 — Bring-up + duplex bake-off (Gate A)

- [ ] **Issue #1:** aarch64 environment (container, PyTorch/transformers ARM,
      vLLM or llama.cpp fallback) on one node
- [ ] **Issue #4:** PersonaPlex 7B + MiniCPM-o 4.5 demos; measure turn-taking
      latency, barge-in, speak/listen control; record baseline sessions;
      read MiniCPM-o's TDM/interleaved-decoder code (Layer-A reference)
- [ ] **Issue #5:** delegation boundary v0 — `<delegate>` → async RPC → Kimi
      API → streamed re-integration (works with unmodified base model)
- [ ] **Issue #2:** tick-grid tokenizer/streamer (80 ms) + Layer-A driver spec
      finalized against MiniCPM-o findings
- [ ] **Issue #3:** synthetic factory v0 starts producing grid-aligned data

**Gate A:** recorded duplex demo with live delegation + baseline metrics.

## Week 3–5 — The Duplex Live model (before allocation expires)

- [ ] **Issue #7:** Qwen3-Omni Layer-A implementation + Phase-1 fine-tune
      (frozen backbone, new token embeddings, tick driver)
- [ ] **Issue #6:** eval harness v0 gating every checkpoint
- [ ] Phase-2 full FT if Phase 1 lands early; text replay to protect reasoning
- [ ] Head-to-head: ours vs PersonaPlex vs MiniCPM-o on interactivity;
      vs base Qwen3-Omni on intelligence (no-regression gate)

**Gate B (by 2026-10-21):** our fine-tune matches PersonaPlex interactivity
with clearly higher intelligence → model v1. If not, document the ceiling.

## After the allocation (conditional)

- Scratch 1–3B track only with a new/renewed allocation and a documented
  fine-tune ceiling. Not plannable inside the current window.

## Compute budget vs. plan

| Item | Est. node-hours | % of 4,974 |
|---|---|---|
| Bring-up + smoke tests | 20–50 | ~1% |
| Bake-off demos + baselines | 10–30 | <1% |
| Synthetic factory (TTS + judging on GH200) | 50–200 | ~2–4% |
| Qwen3-Omni Phase-1 FT (4–8 nodes × days) | 100–800 | ~2–16% |
| Phase-2 full FT | 200–1,500 | ~4–30% |
| Eval/iteration slack | remainder | — |

## First actions (this week)

1. Run `runbooks/01-isambard-bringup.md` on Isambard.
2. Run `runbooks/02-personaplex-test.md` (then the MiniCPM-o equivalent).
3. File results back into Issues #1/#4.
