# Qwen Live — code reading notes (2026-09-19)

Source: github.com/QwenLM/qwen-code `packages/qwen-live` @ main (README
671 lines, package.json, LICENSE). Companion to docs/09 (Layer-A) and
docs/04 (serving) — Qwen Live is the strongest open *harness* reference
for the clock/orchestration layer.

## What it is

A standalone Node 22+ TypeScript daemon ("realtime voice control plane")
that connects three parties:

1. **Live Host** — macOS-only Electron overlay app (orb UX, mic/screen
   capture, Command+E) over a versioned WebSocket protocol (v9) +
   `~/.qwen/live/daemon.json` discovery file.
2. **DashScope Realtime API** — `qwen3.5-omni-plus-realtime` /
   `qwen3.5-omni-flash-realtime` "owns the conversation: VAD, direct
   answers, and a tool surface" (session API: PCM 16 kHz in / 24 kHz out,
   OpenAI-Realtime-style events incl. `response.created`; VAD mode is a
   protocol-settable option).
3. **Coding sessions** via BackendAdaptors: `qwen serve` REST/SSE, or any
   **ACP-compatible agent** (`qwen --acp`, gemini, claude, codex…) as a
   child process over JSON-RPC stdio; multiple backends with per-session
   routing.

Plus: local **Memory** libraries (`~/.qwen-live/memories/`, consolidation
via `qwen3.7-plus`), JSONL session logs, and a **Proactive** subsystem
(condition monitors, narration, reminders — each perception task on an
independent text-only realtime connection).

## Open-source verdict

- The daemon + host protocol code: **Apache-2.0** ✅ (real, usable).
- The duplex brain: **closed**. `qwen3.5-omni-*-realtime` is DashScope
  API-only; open-weight status of Qwen3.5-Omni unconfirmed as of the
  March 2026 release coverage (later reports claim open weights for
  follow-on variants — unverified). Even with open base weights, the
  "-realtime" turn-taking/VAD serving configuration is a hosted system,
  not published artifacts. Memory consolidation (`qwen3.7-plus`) also
  API-only.
- So: **open harness, rented capability.** You cannot self-host the
  duplex loop, cannot fine-tune its turn policy, and mic/screen/audio all
  transit Alibaba's cloud.

## Architecture classification

Neither of the two extremes — it is a **harness wrapped around a
natively-duplex hosted model**:

- NOT a cascade of small models (no client ASR→LLM→TTS chain; VAD and
  turn-taking live model-side in the realtime session).
- NOT an open native-duplex model either — the duplex capability is
  Alibaba's closed serving stack. The open code is clock + orchestration
  (sessions, memory, task routing, proactive scheduling, playback).

This is the OpenAI Realtime API pattern with an open client/control plane.

## Deltas vs Duplex Live (ours)

| | Qwen Live | Duplex Live |
|---|---|---|
| Duplex capability | rented (DashScope realtime) | **in open weights** (Qwen3-Omni + Layer-A tick fine-tune) |
| Turn policy | fixed by Alibaba; VAD mode toggle | **trainable** (factory data: yield, barge, backchannel, T5 addressee) |
| Self-host / data | no (cloud) | yes (GH200, all on-prem) |
| Clock/orchestration | mature, shipped | ours to build (docs/04) — their daemon is the reference |
| Research model | — | 30B-A3B MoE, 80 ms grid, <sil>/<yield>/<delegate> tokens |

Their need for a specialized "-realtime" variant is itself a datapoint
FOR our thesis: duplex turn-taking is a trained-in capability, not an
emergent one — matches our bake-off (both omni references failed T4/T5).

## Learnings to take (ranked)

1. **Demo harness shortcut**: the daemon's model side is a websocket
   protocol. Serving our Layer-A model behind a DashScope-realtime-
   compatible endpoint makes qwen-live (Host orb + daemon) a free,
   polished demo client for our model. Build the adapter when Layer-A
   serves (Issue #7 → demo).
2. **Proactive announcement protocol**: monitor events wait for foreground
   speech to finish, play FIFO, per-event ACK retires only that event,
   cooldowns + false-edge rules. This is the systems-side twin of our
   timed_task/delegation policy data — copy the semantics for our clock
   layer and as scenario shapes in the factory.
3. **ACP as the delegation boundary**: multi-backend agent routing over
   JSON-RPC stdio is a cleaner Issue #5 design than a Kimi-specific
   adapter — speak ACP, get every coding agent for free.
4. **Sessions outlive calls**; daemon owns the live session log; backend
   sessions keep running after hangup. Matches our streaming-session
   design; their JSONL layout is a good schema to borrow.
5. **Diagnostics**: frameHash (first-16 SHA256 of JPEG) threaded across
   capture→send→ack; per-commit counters; credential-redacted logs;
   bounded debug recordings. Issue #6's eval harness should steal this
   traceability pattern wholesale.
6. **Audio plumbing war stories**: 16k-in/24k-out, output-end-marker
   negotiation, continuous band-limited resampling scheduled at integer
   sample boundaries (avoids chunk-conversion spikes). Directly
   applicable to our serving layer (24 kHz canvas already matches).
7. **Privacy posture**: memory retains dialogue + facts, never raw mic
   audio or frames; recordings are opt-in debug with retention bounds.
8. **Voice-agent permission UX**: Allow/Deny surfaced per-request from
   the task panel, "never invents an approval," silent text context
   queued while the model is busy. Safety-relevant for our live demo.

Do NOT copy: the closed-brain dependency (mic/screen to cloud) — the
opposite of our self-hosted constraint; and their macOS-only Host.

## Watch item

If Qwen3.5-Omni (or successor) ships open weights with realtime-capable
serving code, it becomes a candidate Layer-A base (newer than our
Qwen3-Omni-30B). Check quarterly; the swap cost is low because Layer-A
is spec'd against the architecture, not the checkpoint.
