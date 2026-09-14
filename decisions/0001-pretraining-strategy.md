# ADR-0001: Pretraining Strategy — Staged Hybrid (lab-scale)

**Status:** Superseded by ADR-0002 (2026-09-14). Preserved for history.

Original plan assumed a funded commercial lab: Phase A duplex adaptation on an
open-source base ($1.5–5M), Phase B from-scratch 30–80B MoE pretraining
($15–60M), gated by interactivity targets.

Reason for supersession: program rescoped to a mini open-source effort on free
Isambard-AI compute; no commercial-scale pretraining budget. The core principle
survives unchanged: duplex/live capability is trained into the model, never
bolted on via a VAD/turn-detector harness.
