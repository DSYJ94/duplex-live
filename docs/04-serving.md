# 04 — Serving: Stateful Streaming Inference

Adapted from GPT-Live's production lessons: **the voice must flow** — every other concern is subordinate.

## Principles

1. Media path is separate from application logic. Audio/video flows client ⇄ media frontend ⇄ model on a dedicated fast path; tools, delegation, persistence, analytics live behind an async RPC boundary.
2. The system capacity unit is **concurrent sessions with every frame on schedule**, not GPU throughput. CPU-side stream handlers, queues, and network paths scale with inference.
3. Every delay — transport, processing, inference — is an audible artifact. Budget latency end-to-end, per 200 ms micro-turn.

## Streaming sessions

- Client sends each ~200 ms chunk as a request; the server **appends into a persistent sequence in GPU memory** — no reallocation or metadata recompute per micro-turn (TML upstreamed this pattern to SGLang; adopt/contribute).
- Kernels optimized for frequent small prefills + small decodes: gather+GEMV for MoE instead of grouped GEMM; bidirectional-shape-aware kernels.
- Language of implementation for media frontend + inference glue: Go or Rust (GPT-Live's Go rewrite moved p95 to the old system's p50); Python only in training/research.

## Long sessions: handoff and compaction

- **Seamless instance handoff:** warm a replacement instance alongside the live one, prefill with session context, run both in parallel, cut over when ready. Used for scaling events, failures, and upgrades.
- **Dynamic compaction as a managed transition:** when context nears the limit, compact while the original instance keeps chatting, prefill a replacement with compacted context, cut over with zero media interruption. Compaction invalidates the KV cache — never do it on the live instance.

## Delegation path (serving side)

- Background-model session created and prefilled at conversation start; stable session affinity; prompt caching.
- Latency levers: reasoning effort caps, output limits, tool-schema sizing, minimized model↔tool round trips. Full delegation loop has a responsiveness budget; the core can stall briefly but cannot cover arbitrary slowness.

## Transport & startup

- WebRTC-class transport: survives packet loss, clock drift, reconnects; subtle audio stretch/catch-up for late packets.
- Session-startup on the critical path: minimize handshake round trips (WARP-style collapsed handshakes; pre-negotiated parameters / instant-connect pattern so the first UDP packet can materialize the session).
- Geography is first-order: regional inference capacity + traffic steering; latency dashboards broken down by source geography.

## Production validation method (do not skip)

GPT-Live's **silent/shadow test** pattern is our launch gate:

1. Route a small, growing share of real sessions through the new stack in read-only mode alongside the incumbent.
2. Expose real clients, networks, session lengths, geographies — without changing what users hear.
3. Expect failures that only appear across realistic lifecycles: long-session memory pressure, reconnect races, compaction/restoration bugs, shutdown handshakes.
4. Graduate only with granular telemetry, known-good config validation, staged ramps, and instant path-level kill switches.

## SLO targets (v1)

| Metric | Target |
|---|---|
| First audio out after session start | < 800 ms p50 |
| Turn-taking latency (end of user speech → model speech) | < 600 ms p50, < 1 s p95 |
| Frame-schedule adherence | 99.9% of micro-turns on time |
| Delegation first-useful-result | < 3 s p50 |
| Session handoff / compaction media interruption | 0 audible artifacts |
