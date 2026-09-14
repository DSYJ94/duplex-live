# Runbook 02 — PersonaPlex 7B test (Issue #4, part 1)

Goal: a recorded full-duplex conversation on Isambard + baseline metrics.
PersonaPlex requires CUDA — GH200 qualifies. It will **not** run on Apple
Silicon locally; all testing happens on Isambard.

## 1. Setup (inside the ARM container from Runbook 01)

```bash
git clone https://github.com/NVIDIA/personaplex && cd personaplex
pip install -e .            # pulls moshi-family deps; verify aarch64 wheels build
# weights: follow README — HF hub, accept NVIDIA Open Model License first
# NOTE: license stack = MIT code + NVIDIA OML weights + CC-BY-4.0 Moshi base
#       research/eval use OK; record restrictions in Issue #4
```

## 2. Launch the duplex server

```bash
# flags per repo README — verify against the current README before running
python -m personaplex.server --host 0.0.0.0 --port 8998
# client: repo web UI or the provided client script; tunnel via ssh -L 8998:localhost:8998
```

## 3. Scripted test protocol (record every session)

| # | Test | What to do | Metric |
|---|---|---|---|
| T1 | Baseline turn-taking | Ask 10 short questions | user-end → model-speech latency (target < 1 s) |
| T2 | Barge-in | Interrupt mid-response, 5× | does it yield? time-to-yield |
| T3 | Backchannel | Say "mm-hmm"/"right" while it talks | does it inappropriately stop? |
| T4 | Thinking pause | Pause 2–3 s mid-sentence | premature interruption? |
| T5 | Background speech | Have a second voice talk nearby | false-trigger rate |
| T6 | Long session | 15 min continuous | stability, drift, memory |

Record audio of both channels + timestamps; save under `evals/baselines/personaplex/`.

## 4. MiniCPM-o 4.5 equivalent (part 2)

```bash
pip install transformers librosa
# HF: openbmb/MiniCPM-o-4_5, duplex mode:
#   model = AutoModel.from_pretrained(..., trust_remote_code=True).as_duplex()
#   per-chunk: model.streaming_prefill(...) / model.streaming_generate(...)
#   see github.com/OpenBMB/minicpm-o-4_5-pytorch-simple-demo
```

Run T1–T6 identically. Note its 1 Hz speak/listen control behavior explicitly —
we compare against PersonaPlex's continuous control.

## 5. Code-reading task (feeds docs/09)

While on MiniCPM-o source, extract notes on: TDM time-slice format, interleaved
text–speech token layout, how retraction/yield is implemented. Commit notes to
`docs/notes-minicpm-layera.md`.
