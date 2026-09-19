# Delegation boundary v0 (Issue #5)

The 'brain' seam of docs/01: hard questions delegate across an async
boundary while the duplex media loop keeps running. Layer-A's `<delegate>`
token fires this; results stream back and are woven in.

- `boundary.py` — `DelegationBoundary`: `fire()` returns in microseconds
  (verified 0.1 ms against a 5 s-slow brain); chunks drain via non-blocking
  `poll()` between ticks; pre-warmed resident session; per-delegation
  latency ledger (`first_chunk_s` vs the **< 3 s p50** budget); ADR-0002
  enforced by construction — ledger + outputs live only under
  `runs/delegation/`, flagged `closed_api`, never in training manifests.
- `backends.py` — `BrainBackend` implementations:
  - `MockBackend` (deterministic timing; tests/CI)
  - `KimiBackend` (Moonshot OpenAI-compatible SSE; `KIMI_API_KEY`;
    base_url overridable → self-hosted vLLM/sglang is the open-model swap)
  - `AcpBackend` (any ACP agent — `claude/qwen/codex/gemini --acp` — over
    JSON-RPC stdio; the generic background brain per docs/notes-qwen-live)

```bash
python -m serving.delegation.demo --backend mock
KIMI_API_KEY=… python -m serving.delegation.demo --backend kimi
python -m serving.delegation.demo --backend acp -- <agent> --acp
```

Status: mock validated (fire 317 µs, first chunk 0.81 s, budget report,
ledger rows); **ACP live handshake pending** — the locally installed
claude (2.1.278) and codex builds predate `--acp` support; the client
follows the published ACP spec (initialize → session/new →
session/prompt + session/update streams) and activates when an
ACP-capable agent is present. Kimi path is standard OpenAI-compatible
streaming, untested until a key is provided.

Next: wire `<delegate>` interception into the Layer-A tick loop (Issue
#7); prewarm with real conversation context at session start; latency
budget against a live brain.
