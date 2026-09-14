# Duplex Live — Visualization

`index.html` is a self-contained (zero-dependency) interactive page:

1. **Candidate base-model comparison** — Qwen3-Omni (primary), PersonaPlex (duplex reference), Moshi, GLM-4-Voice, Qwen2.5-Omni, Llama-Omni/Freeze-Omni, DuplexOmni, with licenses, duplex ratings, and program roles.
2. **Animated test-time flow** — press ▶ to run a delegation scenario: user speech → WebRTC transport → media frontend → duplex core (200 ms micro-turn loop) → async delegation boundary → background model (Kimi API / open 32B+) → streamed result woven back into speech. Click any node for its design rationale and latency budget. A ticker shows the token-level view: `[in] [out] [·sil·] [<delegate>] … [⇐result]`.

Open `index.html` directly in any browser, or serve with any static server (e.g. `npx vite`).
