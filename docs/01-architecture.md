# 01 — Architecture

## System overview

```
        ┌─────────────────────── live path (hard real-time) ───────────────────────┐
        │                                                                          │
 user ⇄ │  media transport (WebRTC-class)  ⇄  DUPLEX CORE MODEL (micro-turn loop)  │
        │                                          │                               │
        └──────────────────────────────────────────┼───────────────────────────────┘
                                                    │ async RPC (never blocks media)
                                                    ▼
                                   BACKGROUND REASONING MODEL
                                   (deep reasoning, tools, search, generative UI)
```

Two model principles, both mandatory:

1. **The duplex core owns the conversation.** It listens and speaks simultaneously, decides when to talk, when to stay silent, when to interject, and when to delegate.
2. **The background model is invisible to timing.** A slow tool call delays its own result, never the audio stream.

## Duplex core: time-aligned micro-turns

- Wall-clock time is the organizing axis. Input streams (audio frames, video frames, text) and output streams (speech, text) are chunked into **~200 ms micro-turns**.
- The model's token sequence interleaves input and output micro-turns: `in₀ out₀ in₁ out₁ in₂ out₂ …`. Silence, overlap, interruption, and backchannels are *data in the sequence*, not control signals from a harness.
- No turn boundaries. No VAD. Turn-taking, interjection, and simultaneous speech (e.g. live translation) are learned behaviors.
- **Time-awareness:** elapsed-time is explicit in the representation (micro-turn index / timing tokens), enabling "remind me every 4 seconds"-class behaviors.

## Modality pathways (encoder-free early fusion — scratch track)

| Stream | In | Out |
|---|---|---|
| Audio | raw waveform → **dMel** features → lightweight learned embedding (no Whisper-style pretrained encoder) | transformer hidden states → **flow-matching head** → mel → waveform |
| Video | frames → 40×40 patches → **hMLP** patch encoder | (v0: none; later: generative UI via delegation) |
| Text | standard tokenizer | standard unembedding |

All pathways are **co-trained with the transformer from scratch** (Phase B) or grafted onto the open base via continued pretraining (Phase A). Rationale: frozen external encoders/encoders become interactivity bottlenecks and latency floors.

## Backbone

- MoE transformer; first scratch scale point 30–80B total / 3–8B active (serving-latency bound, see `04-serving.md`).
- Bidirectional (full-duplex) attention pattern over the interleaved micro-turn stream; causal in time.
- Context budget: streaming audio+video accumulates fast (~micro-turn every 200 ms ⇒ 18k micro-turns/hour); context management is a first-class architecture concern (see serving doc).

## Delegation protocol (live adaptation)

Following GPT-Live's delegation design and OpenAI's live-delegation guide:

- **Pre-warmed sessions:** at conversation start, the background model session is created and prefilled with initial context; stable session affinity + prompt caching for subsequent delegations.
- **Rich context package:** the duplex core delegates with the full (compacted) conversation state, not a standalone query.
- **Streamed re-integration:** results stream back; the core decides *when* to weave them into speech — mid-sentence if urgent, at the next natural beat otherwise. The core can stall gracefully ("let me check that—") but cannot hide arbitrarily slow work: the full delegation loop has a latency budget (target: first useful result < 3 s p50).
- **Discrete-view derivation:** a segmentation layer derives provisional→final user/assistant messages from the continuous stream for logging, analytics, safety, and UI — without imposing turns on the model.

## What the model must learn (capabilities checklist)

- [ ] Implicit dialog management (thinking vs. yielding vs. self-correction vs. inviting response)
- [ ] Interruption handling (user barges in → model yields appropriately, not instantly, not never)
- [ ] Backchanneling ("mm-hmm", "right") without stealing the floor
- [ ] Proactive interjection on request ("correct me when I'm wrong")
- [ ] Simultaneous speech (live translation, live commentary)
- [ ] Time-aware action (timed reminders, pace estimation)
- [ ] Visual proactivity ("count my reps", "tell me when X appears")
- [ ] Delegation decisions (when deep thought is worth a pause)
