# Duplex Live

A training program for a **proprietary full-duplex, time-aware interaction model** — the same model class as OpenAI's GPT-Live and Thinking Machines Lab's interaction models.

**North star:** a model that natively listens, watches, thinks, and speaks *at the same time*, with no turn detector in the audio path, and that delegates deep reasoning to a background frontier model without breaking conversational flow.

## Why not a harness?

Turn-based voice stacks (ASR → LLM → TTS, VAD-gated) bolt interactivity on with external scaffolding. That caps quality: the VAD is dumber than the model, interruptions are fake, and proactivity is impossible. Both reference systems — GPT-Live and TML-Interaction — moved interactivity *into the model* so it scales with intelligence. Duplex Live follows that principle: **duplex and live adaptation are trained, not scripted.**

## Program pillars

1. **Full-duplex core model** — time-aligned micro-turns (~200 ms) interleaving input and output across audio, video, and text; no turn boundaries, no VAD in the live path.
2. **Live delegation** — an asynchronous background reasoning/tool model that the duplex core consults mid-conversation, with rich shared context and streamed results.
3. **Stateful streaming serving** — persistent GPU-resident sessions, seamless instance handoff, dynamic context compaction, WebRTC-class media transport.
4. **Interaction-first evaluation** — turn-taking latency, FD-bench-style interactivity, time-awareness, simultaneous speech, visual proactivity — alongside standard intelligence benchmarks.

## The key decision: pretrain from scratch vs. build on open source

Full analysis in [`docs/00-pretraining-decision.md`](docs/00-pretraining-decision.md). **Recommendation: staged hybrid.**

- **Phase A (now):** build the duplex/live stack on an open-source base LLM (text backbone) + from-scratch audio/video pathways. Fastest path to a working duplex model; de-risks micro-turn training, serving, and evals.
- **Phase B (parallel research track):** from-scratch multimodal pretraining with encoder-free early fusion (dMel audio in, flow-head audio out, patch-MLP video), following the TML recipe. Promote to the primary track only when Phase A validates the data pipeline, evals, and serving stack.

Scratch pretraining everything on day one spends 9–12 months and tens of millions of dollars before we learn whether our duplex data recipe works. The hybrid gets a shippable duplex model in ~2 quarters while building the scratch capability we ultimately want for a proprietary model.

## Repository map

| Doc | Contents |
|---|---|
| [docs/00-pretraining-decision.md](docs/00-pretraining-decision.md) | Scratch vs. open-source analysis, cost model, decision gates |
| [docs/01-architecture.md](docs/01-architecture.md) | Model architecture: micro-turns, early fusion, delegation protocol |
| [docs/02-data.md](docs/02-data.md) | Data strategy: duplex conversation data, synthetic generation, mixtures |
| [docs/03-training-program.md](docs/03-training-program.md) | Stages, compute plan, trainer–sampler alignment, checkpoints |
| [docs/04-serving.md](docs/04-serving.md) | Stateful streaming inference, handoff/compaction, media transport |
| [docs/05-evaluation.md](docs/05-evaluation.md) | Interactivity + intelligence benchmark suite and targets |
| [docs/06-safety.md](docs/06-safety.md) | Real-time safety: spoken refusals, red-teaming, long-session robustness |
| [docs/07-roadmap.md](docs/07-roadmap.md) | Milestones, staffing, budget, go/no-go gates |

## References

- OpenAI — *Continuous voice interaction with GPT-Live* (stateful inference, delegation, WARP/Instant Connect, shadow testing)
- OpenAI — Realtime API guides: *Live* and *Live delegation*
- Thinking Machines Lab — *Interaction Models: A Scalable Approach to Human-AI Collaboration* (micro-turns, encoder-free early fusion, streaming sessions, benchmarks)
- Prior art: Moshi, PersonaPlex, Nemotron VoiceChat, Qwen-Omni, FD-bench, Audio MultiChallenge, VoiceBench
