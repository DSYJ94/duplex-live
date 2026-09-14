# data/

Data pipelines for Duplex Live (see `docs/02-data.md`).

- `alignment/` — forced alignment, overlap-aware diarization, micro-turn grid rendering, QC
- `synthetic/` — interaction-dynamics factory (Q1 critical path): scripted interruption/silence/timing scenarios -> open TTS (Qwen3-TTS / Chatterbox class) -> 200 ms grid -> judge filtering
- `video/` — video-grounded stream preparation (action counting, temporal callouts)
- `mixtures/` — mixture configs + contamination checks against eval sets

Rule: closed-API outputs never enter the training corpus (ADR-0002).
