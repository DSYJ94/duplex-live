# M2 — the live loop (Track B plan, 2026-09-28)

Owner goal: by ~2026-10-14, audible correct turn-taking through this
loop. Decode config it will serve: **dense-pt3 + activity gate2 @0.5 +
user-speech mask (hold 3)** — near-perfect onsets (16/17 exact on
trained lessons), known between-turns murmur; Layer-A training
continues in parallel (gate round 3) and swaps in if it beats this.

## The loop, end to end

    mic (80 ms chunks) → ear features → tick loop → voice → speaker
         local Mac         cluster        cluster      cluster
                                          (parity-proven)  (piece bank)

## What already exists

- **Tick loop parity (milestone 1, PASS)**: dry-vs-paced, 0 mismatches,
  the clock does not change what the model says (m2/parity.py).
- **Serving decode**: gate + user-mask wired into parity (M2.2a) with
  the exact freerun precedence; arm 1 saves picks.json = THE reference
  the live loop must match tick-for-tick (docs/04 shadow-test pattern).
- **Voice**: render_model_audio synthesizes each unique word once and
  reuses it (MiniCPM voice-clone). Key realization: the model can only
  emit pieces from its training vocabulary — a FINITE set. Pre-
  synthesizing the whole piece bank offline makes live TTS pure
  concatenation, ~zero latency, no per-word synthesis stall.
- **SLOs (docs/04 v1)**: turn-taking < 600 ms p50; 99.9% of ticks on
  time; first audio < 800 ms.

## Known walls (measured or named)

1. **Piece budget**: while speaking, a word drags up to 8 sequential
   forwards inside one 80 ms tick (~57 ms/forward at pt1). Fix is
   scheduling, not model: the DECISION must be on time; a word's
   pieces may complete in arrears (audio starts tens of ms late —
   inaudible under ~150 ms). Barge-in mid-word: check the user mask
   before each piece (small decode change).
2. **Live ear**: shards carry precomputed features; live must encode
   the mic stream. The encoder works per 4 s window (52 features),
   which runs ahead of the 80 ms tick loop naturally; its per-window
   latency is unmeasured (M2.3).
3. **Transport**: Mac ⇄ GH200 stream; start with the simplest thing
   that survives the demo (websocket over ssh tunnel), upgrade later
   per docs/04.

## Milestones

| # | What | Done when |
|---|---|---|
| 2a | parity + latency on the serving config; eager/compile kernel bench (**job 6923598**) | gates PASS; picks-m2-serving saved; kernel picked |
| 2b | piece budget: decisions on time, pieces in arrears; barge-in check per piece | paced pass holds 99.9% adherence DURING speech; audible word starts unharmed |
| 2c | piece bank: offline synth of every reachable piece; concatenation player | any decode-emittable utterance renders without live synthesis |
| 3 | live ear: streaming 4 s-window encoding feeding the tick loop | shadow loop matches picks-m2-serving with 0 mismatches |
| 4 | end-to-end on cluster, synthetic mic (a lesson's user stem played live) | session audible in the gallery; turn-taking SLO met on scripted audio |
| 5 | real mic + speaker, gallery-recorded | Daryl hears correct turn-taking (the owner goal) |

Risk order: 2b (scheduling) < 2c (bank build is bounded compute) <
3 (encoder latency) < transport glue. Nothing here touches the model
weights — the parity reference keeps the live loop honest at every
step: WHAT is said may never depend on HOW fast we say it.
