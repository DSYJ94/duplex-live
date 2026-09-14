# 08 — Fine-Tuning Duplex into a Turn-Oriented Omni Model

How Qwen3-Omni (Thinker-Talker, turn-oriented) becomes Duplex Live. The key
insight: **turns are a data statistic, not an architectural constraint.**
Nothing in the transformer enforces turn-taking — the training corpus did.
Precedents: Moshi (Helium → full duplex), DuplexOmni (Qwen-family → duplex),
PersonaPlex (Moshi → role-controlled duplex).

## Step 1 — Reformat data into interleaved micro-turn streams

Keep Qwen-Omni's audio token scheme, but construct training sequences where
user-audio tokens and assistant-output tokens interleave on a fixed ~200 ms
grid:

```
[user: hi how are ] [asst: <sil> ] [user: you— actually ] [asst: I'm good— ] ...
  in₀                out₀            in₁                  out₁
```

instead of strict `[user turn][assistant turn]` alternation. Silence, overlap,
hesitation and interruption become data in the sequence.

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

1. **Phase 1 (cheap):** freeze most of the Thinker; train audio pathways +
   new token embeddings only, on synthetic duplex data. Days on a few GH200s.
2. **Phase 2:** full fine-tune with text replay in the mixture so reasoning
   doesn't regress. Checkpoint-gated: no intelligence regression > 2%,
   monotonic interactivity improvement.

## Step 6 — Change the serving loop

At inference, feed a 200 ms chunk every 200 ms **regardless of whether the
user "finished speaking"**. At each step the model samples from {speak, `<sil>`,
`<delegate>`} from its own output distribution. No VAD anywhere in the path —
the model's silence probability is the dialog manager.

## Latency budget check (GH200, 30B-A3B MoE)

~3B active params/token → per-200ms-step decode is small; the binding
constraints are per-step overhead (use persistent streaming sessions,
docs/04) and aarch64 kernel maturity (verify week 1, ADR-0002).

## What we fine-tune vs. what stays

| Component | Action |
|---|---|
| Thinker text/reasoning weights | Freeze in Phase 1, full FT + replay in Phase 2 |
| Audio encoder/talker | Continue training on duplex streams |
| New control/timing embeddings | Train from scratch |
| Vocabulary dims, d_model, layers | Fixed (acceptable — see analysis in conversation) |
| Delegation | Pure serving-layer + `<delegate>` behavior training; no architecture change |
