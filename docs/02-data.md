# 02 — Data Strategy

The data problem is the program's real moat. Duplex interaction data — time-aligned, overlapping, multi-modal — barely exists in the wild at scale.

## The PersonaPlex recipe (validated template — refine from docs/notes)

PersonaPlex (NVIDIA, Jan 2026) shows the two-tier structure that works at our scale:

1. **Real conversational speech:** Fisher English corpus — ~2,000 h of natural telephone conversations (LDC-licensed; check our institutional access via Bristol). Provides genuine turn dynamics, overlap, backchannels.
2. **Synthetic assistant-role conversations:** scripted persona/task dialogs rendered with open TTS (they used Chatterbox/Tortoise). Provides coverage of assistant-side behaviors real corpora lack (helpfulness patterns, instruction following, refusals).

We adopt the same two tiers, plus a third for our novel capabilities:

3. **Interaction-dynamics + delegation scenarios** (our addition): interruption taxonomy, silence semantics, simultaneous speech, timed tasks, `<delegate>` scenarios, speech retraction — fully synthetic, fully controllable.

## Data layers

### Layer 1: Intelligence foundation
- Inherited from the base model (Qwen3-Omni / MiniCPM-o). We add text replay only, to prevent regression during duplex fine-tuning.

### Layer 2: Duplex conversation corpus (real)

| Source | Volume | Duplex quality | Access |
|---|---|---|---|
| **Fisher English** (telephone conversations) | ~2,000 h | High (natural overlap/backchannels) | LDC license — check Bristol institutional access; do not pirate |
| Podcasts / interviews (licensed) | Medium | Medium-high | licensing work |
| Meetings / panels | Medium | High overlap, harder attribution | later |
| Sessions with our own demos (PersonaPlex/MiniCPM-o bake-off recordings) | Growing | On-distribution | free, starts Week 1–2 (Issue #4 recordings!) |

**Alignment requirement:** every stream micro-turn/tick aligned (80 ms grid), precise timestamps, overlap preserved — not segmented into turns. The alignment toolchain (Issue #2/#3) gates everything.

### Layer 3: Synthetic interaction dynamics (the scaling lever)

- **Assistant-role conversations** (PersonaPlex-style): persona + task scripts → multi-voice open TTS (Qwen3-TTS / Chatterbox class) → naturalistic prosody.
- **Interruption taxonomy:** barge-in to correct / change topic / to-ignore; collaborative overlap; backchannels.
- **Silence semantics:** thinking pause vs. end-of-thought vs. inviting response.
- **Simultaneous speech:** live translation, commentary, pronunciation correction.
- **Time-aware tasks:** timers, pacing, "every N seconds" instructions.
- **Delegation scenarios:** conversations where delegating is correct, with graceful stalls and streamed re-integration examples.
- **Speech retraction:** user barges in mid-assistant-speech; clean yield + resume/revise.
- **Refusal/over-refusal speech data** via TTS over text refusal distributions (docs/06).

Pipeline: script scenarios → TTS with overlap → render onto 80 ms grid → judge filter. **Rule: no closed-API outputs in training data (ADR-0002).**

### Layer 4: Video-grounded streams (later; MiniCPM-o/Qwen3-Omni both support vision)

- Action counting, temporal start/stop callouts, proactive-moment QA.

## Mixture plan (Phase A duplex fine-tune)

| Component | Share (tokens) |
|---|---|
| Real duplex conversations (Fisher + licensed) | 35% |
| Synthetic assistant-role + interaction dynamics | 25% |
| Text intelligence replay | 25% |
| Video-grounded streams | 10% |
| Safety/refusal speech | 5% |

## Governance

- Provenance tracking per stream; license tag in metadata.
- PII scrubbing on recorded audio; voice-consent registry.
- Dedup against eval sets every drop.

## References

- PersonaPlex training breakdown video (Fisher + synthetic assistant roles): https://www.youtube.com/watch?v=n_m0fqp8xwQ
- NVIDIA PersonaPlex page: https://research.nvidia.com/labs/adlr/personaplex/
