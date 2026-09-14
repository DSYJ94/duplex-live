# serving/

Stateful streaming inference, mini scale (see `docs/04-serving.md`).

- `session/` — persistent streaming sessions (append-per-200ms-chunk)
- `frontend/` — media frontend (WebRTC-class transport)
- `delegation/` — async RPC boundary to background model (Kimi API first, self-hosted open model later); pre-warmed session, <3 s first-useful-result budget
- `segmentation/` — provisional->final discrete-turn derivation for UI/logging/safety

Mini target: single GH200 node serving one duplex session end-to-end.
