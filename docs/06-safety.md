# 06 — Safety for Real-Time Interaction

Real-time speech stresses safety differently from turn-based text: refusals must be colloquial and instant, and robustness must hold across hour-long spoken sessions.

## Axes (following TML's framing)

### 1. Modality-appropriate refusals
- Generate refusal and **over-refusal** training data with TTS across the disallowed-topic taxonomy; refusal phrasing must be natural spoken language, firm but conversational.
- Calibrate the boundary explicitly: track both under-refusal and over-refusal rates in every checkpoint report.
- Maintain close behavioral parity with text-modality refusals — the duplex model must not become the safety bypass for the text model.

### 2. Long-horizon robustness
- Automated red-teaming harness generating **multi-turn spoken** adversarial sessions (slow-burn jailbreaks, topic drift, emotional escalation, voice-based social engineering).
- Soak-test safety: refusal stability measured at minute 1, 10, 30, 60 of continuous sessions.

## Duplex-specific risks

| Risk | Mitigation |
|---|---|
| Interruption used to derail safety state mid-sentence | Safety state is part of persistent session context, not per-turn; red-team barge-in jailbreaks |
| Background-model results woven into speech without re-check | All delegated output passes the same spoken-safety filter before vocalization |
| Voice cloning / impersonation requests | Strict voice-provenance policy; refusal data covers it; watermarking on output audio |
| Always-listening perception / privacy | Clear session UX, on-device wake gating where feasible, data-retention policy in product spec |
| Latency pressure eroding safety checks | Safety classifiers run on the streaming path with their own latency budget (< 50 ms); never skipped under load — degrade gracefully instead |

## Process

- Pre-launch: external red-team + shadow-test safety telemetry (`04-serving.md`).
- Post-launch: conversation-level safety analytics on the *authoritative* transcript view (not the speculative one).
- Incident response: path-level kill switches; safety config hot-swappable without session interruption.
