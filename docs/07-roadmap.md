# 07 — Roadmap, Staffing, Budget

## Milestones

### Q1 — Foundations
- [ ] Micro-turn tokenizer/streamer + alignment data toolchain
- [ ] Base model selection (license, MoE, latency benchmarks) — **Decision D1**
- [ ] Eval harness v0 (FD-bench-style + latency simulation) in CI
- [ ] Duplex CPT run #1 on open base; first simultaneous listen/speak demo
- [ ] Serving: streaming-session prototype (persistent GPU sequence)

### Q2 — Phase A complete → **Gate 1**
- [ ] Duplex SFT + preference training; delegation protocol with a frontier background model
- [ ] Serving: handoff + compaction without media interruption; WebRTC-class transport
- [ ] Internal interactivity benchmarks (TimeSpeak/CueSpeak/RepCount-style) live
- [ ] v0 product demo; human eval round 1
- [ ] **Gate 1 review:** latency < 600 ms p50, interactivity ≥ 60, 10-min stable sessions → fund Phase B

### Q3–Q4 — Scratch track spin-up
- [ ] 1–3B scaling-law runs; scratch mixture fixed → **Gate 2**
- [ ] Batch-invariant kernels + trainer–sampler bitwise alignment
- [ ] Synthetic interaction data factory at full throughput

### Q5–Q8 — Phase B main run
- [ ] Main scratch pretraining (30–80B MoE / 3–8B active)
- [ ] Post-training + safety battery; shadow test against Phase A model
- [ ] **Gate 3:** scratch model ≥ Phase A model on interactivity, ≥ on intelligence → cut over

## Team (core, ~14–20)

| Area | Headcount |
|---|---|
| Pretraining / architecture | 4–6 |
| Data (alignment toolchain, synthetic factory, licensing) | 3–4 |
| Serving / infrastructure (streaming sessions, transport, kernels) | 4–5 |
| Eval & safety | 2–3 |
| Product / UX (session UX, discrete-view derivation) | 1–2 |

## Budget shape (order-of-magnitude)

| Item | Phase A | Phase B |
|---|---|---|
| Compute | $1.5M–$5M | $15M–$60M (main run + scaling runs) |
| Data licensing & collection | $0.5M–$2M | $1M–$3M |
| Team (annualized) | dominant cost either way | — |

## Go / No-Go summary

| Gate | Question | If NO |
|---|---|---|
| D1 (Q1) | Which open base? | Re-benchmark candidates |
| Gate 1 (Q2) | Does duplex adaptation hit interactivity targets? | Fix data/training; do **not** start scratch run |
| Gate 2 (Q4) | Is the scratch mixture validated at small scale? | Iterate at 1–3B |
| Gate 3 (Q8) | Does the scratch model beat the adapted one? | Ship Phase A model; re-plan Phase B scale |

## Immediate next actions

1. Stand up repo scaffolding: `tokenizer/`, `data/`, `training/`, `serving/`, `evals/`.
2. Open tracking issues for each Q1 milestone.
3. Base-model bake-off: shortlist 3 candidates, run latency + intelligence benchmarks, record D1 as an ADR.
