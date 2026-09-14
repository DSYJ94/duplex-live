# 08 — Fine-Tuning Duplex into a Turn-Oriented Omni Model

How Qwen3-Omni (Thinker-Talker, turn-oriented) becomes Duplex Live.

## First, get the layers right (corrected 2026-09-14)

"Full duplex" is **three separable layers**, not one:

| Layer | What it is | Architectural or data? |
|---|---|---|
| **A. Capability** | The model can ingest new input *while* generating output | **Architectural** — sequence format + inference loop. A vanilla LLM structurally cannot do this: during generation there is no input channel |
| **B. Policy** | When to speak, stay silent, backchannel, barge in | **Data** — learned from overlap/silence/interruption statistics in training |
| **C. Clock** | Serving loop that feeds 200 ms chunks unconditionally with persistent GPU state | **Systems** — pure engineering |

OpenAI's GPT-Live post says "full-duplex architecture" and means Layer A:
"continuously processes input while generating output… makes interaction
decisions many times per second." TML's micro-turn interleaving is the same
layer. **Architecture provides the channel, data provides the manners,
serving provides the clock.**

Qwen3-Omni today is **half-duplex streaming**: low-latency streaming in and
out, but it does not interleave new user audio into its context while the
Talker is mid-generation — perception freezes during a response. So all three
layers need work: a surgical Layer-A modification (days-to-weeks, not
pretraining-scale), Layer-B training data, and the Layer-C session loop.

Moshi is the existence proof that Layers A+B are achievable at fine-tuning
scale on a turn-based base (Helium 7B). DuplexOmni did the same on a
Qwen-family base.

## Step 0 — Layer A: interleaved input during generation (architectural)

Modify the model's input pipeline so user audio tokens are appended into the
context **continuously, every ~200 ms, including mid-generation**:

- Interleaved sequence layout: `in₀ out₀ in₁ out₁ …` as the native context
  format (not block-structured turns).
- Attention mask + KV-cache handling that admits new input tokens between
  output tokens.
- Talker consumes Thinker hidden states on the same grid rather than once per
  turn.

This is the true "full-duplex architecture" change. Everything below trains
the behavior on top of it.

## Step 1 — Training data on the micro-turn grid (Layer B)

Keep Qwen-Omni's audio token scheme, but construct training sequences where
user-audio tokens and assistant-output tokens interleave on the fixed 200 ms
grid. Silence, overlap, hesitation and interruption become data in the
sequence.

## Step 2 — Extend the vocabulary (new embeddings, trained from scratch)

| Token | Meaning |
|---|---|
| `<sil>` | emit nothing this micro-turn — **this is the learned turn detector** |
| `<t+Ns>` | timing tokens: elapsed-time buckets, enabling time-aware behavior |
| `<delegate>` | trigger async handoff; followed by a structured context package |
| `<yield>` / `<barge-in>` | interruption semantics markers |

## Step 3 — Losses that teach timing, not just content

- Next-token prediction over the interleaved stream.
- **Up-weighted loss on silence/timing decisions** — the model must learn when
  NOT to speak.
- Primary failure mode: collapse back to turn-taking. Countermeasures: heavy
  overlap/silence share in the mixture (see docs/02), timing-prediction loss
  term, eval-gating on simultaneous-speech tasks (docs/05).

## Step 4 — Teach the Talker interruption

Train on **truncated outputs**: sequences where the user barges in
mid-assistant-speech; the Talker must stop cleanly, then resume or revise
("speech retraction"). This data comes from the synthetic factory — scripted
barge-in scenarios rendered with open TTS (Qwen3-TTS / Chatterbox class).

## Step 5 — Two-phase training

1. **Phase 1 (cheap):** freeze most of the Thinker; train the Layer-A
   machinery, audio pathways + new token embeddings only, on synthetic duplex
   data. Days on a few GH200s.
2. **Phase 2:** full fine-tune with text replay in the mixture so reasoning
   doesn't regress. Checkpoint-gated: no intelligence regression > 2%,
   monotonic interactivity improvement.

## Step 6 — Layer C: the serving loop

At inference, feed a 200 ms chunk every 200 ms **regardless of whether the
user "finished speaking"**. At each step the model samples from {speak, `<sil>`,
`<delegate>`} from its own output distribution. No VAD anywhere in the path —
the model's silence probability is the dialog manager. Persistent GPU-resident
session state (docs/04).

## Latency budget check (GH200, 30B-A3B MoE)

~3B active params/token → per-200ms-step decode is small; the binding
constraints are per-step overhead (use persistent streaming sessions,
docs/04) and aarch64 kernel maturity (verify week 1, ADR-0002).

## What we change vs. what stays

| Component | Action |
|---|---|
| Input pipeline / sequence format | **Modify (Layer A)** — interleaved input during generation |
| Thinker text/reasoning weights | Freeze in Phase 1, full FT + replay in Phase 2 |
| Audio encoder/talker | Continue training on duplex streams |
| New control/timing embeddings | Train from scratch |
| d_model, layer count, attention internals | Fixed (acceptable) |
| Delegation | Pure serving-layer + `<delegate>` behavior training; no architecture change |
