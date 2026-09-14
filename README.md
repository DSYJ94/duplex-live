# Duplex Live

A **mini open-source program** to build a full-duplex, time-aware interaction model — the model class of OpenAI's GPT-Live and Thinking Machines Lab's interaction models — at research scale on **Isambard-AI** (GH200 Grace Hopper, aarch64), not commercial-lab scale.

**North star:** a model that listens, thinks, and speaks *at the same time* — no turn detector in the audio path — and that delegates deep reasoning to a smarter background model (Kimi API or a self-hosted open model) without breaking conversational flow.

## Strategy (current — see decisions/)

**Sequential fine-tune → gated scratch**, at mini scale:

1. **Fine-tune track (primary):** Qwen3-Omni 30B-A3B (Apache-2.0, highest intelligence per latency) fine-tuned into a full-duplex, time-aware model via micro-turn data reformatting + new control/timing tokens. PersonaPlex 7B is the duplex-behavior reference and quick demo path.
2. **Delegation (from day one):** async RPC boundary to Kimi API / open 32B+ background model — systems work, not model scale.
3. **Scratch track (gated):** 1–3B from-scratch time-aware duplex model on Isambard (~30–80 GH200-days), only if the fine-tune track hits a demonstrated interactivity ceiling. This is the potential research-paper contribution.

ADR-0001 (staged hybrid, lab-scale) is preserved for history; **ADR-0002 is the current plan of record**.

## Repository map

| Path | Contents |
|---|---|
| [docs/00-pretraining-decision.md](docs/00-pretraining-decision.md) | Original scratch-vs-open analysis (lab-scale context) |
| [docs/01-architecture.md](docs/01-architecture.md) | Micro-turns, early fusion, delegation protocol (scale-agnostic) |
| [docs/02-data.md](docs/02-data.md) | Duplex data strategy; synthetic interaction factory |
| [docs/03-training-program.md](docs/03-training-program.md) | Training stages; trainer–sampler alignment |
| [docs/04-serving.md](docs/04-serving.md) | Streaming sessions, handoff/compaction, transport |
| [docs/05-evaluation.md](docs/05-evaluation.md) | Interactivity + intelligence benchmark suite |
| [docs/06-safety.md](docs/06-safety.md) | Real-time safety |
| [docs/07-roadmap.md](docs/07-roadmap.md) | **Mini roadmap (Isambard-scale, 2–3 quarters)** |
| [docs/08-duplex-finetune-recipe.md](docs/08-duplex-finetune-recipe.md) | How to fine-tune duplex into turn-oriented Qwen3-Omni |
| [decisions/](decisions/) | ADRs — 0001 (superseded), 0002 (current) |
| [viz/index.html](viz/index.html) | Interactive candidate comparison + animated test-time flow visualization (open in any browser) |
| tokenizer/ · data/ · training/ · serving/ · evals/ | Code scaffold (see each README) |

## Hardware note

Isambard-AI is **aarch64 (ARM)**: all containers/environments must be ARM-native; x86 Conda envs and binaries will not run.

## References

- OpenAI — *Continuous voice interaction with GPT-Live*; Realtime API guides (Live, Live delegation)
- Thinking Machines Lab — *Interaction Models* (May 2026)
- NVIDIA — *PersonaPlex* (Jan 2026); Kyutai — *Moshi* (2024); *DuplexOmni* (arXiv, June 2026)
