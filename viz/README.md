# Duplex Live — Visualization

`index.html` is a self-contained (zero-dependency) interactive page:

1. **Candidate base-model comparison** — Qwen3-Omni (primary), PersonaPlex (duplex reference), Moshi, GLM-4-Voice, Qwen2.5-Omni, Llama-Omni/Freeze-Omni, DuplexOmni, with licenses, duplex ratings, and program roles.
2. **Animated test-time flow** — press ▶ to run a delegation scenario: user speech → WebRTC transport → media frontend → duplex core (200 ms micro-turn loop) → async delegation boundary → background model (Kimi API / open 32B+) → streamed result woven back into speech. Click any node for its design rationale and latency budget. A ticker shows the token-level view: `[in] [out] [·sil·] [<delegate>] … [⇐result]`.

Open `index.html` directly in any browser, or serve with any static server (e.g. `npx vite`).

## `listen.py` — localhost listening gallery (audio, not architecture)

Zero-dependency (stdlib) server for auditioning everything the pipeline
produces with a voice in it: rendered batches (mixed / stems / every
utterance joined with its WER + what the ASR judge heard) and the voice
pool (each reference clip + per-voice WER from the filter reports).

```bash
python -m viz.listen              # -> http://localhost:8765
```

Reads local copies under `runs/` — sync from Isambard after each step
(rsync recipes in the `listen.py` docstring) and refresh the page. This is
the standing audit loop for the user: every render/filter/voice step ends
with a sync + "new audio is up" ping, never a bare path on a cluster.
