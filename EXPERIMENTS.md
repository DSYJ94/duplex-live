# Phase-1 ablation ledger

One row per training run: the recipe, the held-out exam result, the
verdict. **Keep this current with every run** — it is the memory of what
we tried so we never re-buy the same lesson. (Plain-language rule
applies: acc = how often the model's top pick was the right token; prob =
how much of its confidence it put on the right token, on average.)

**What "the exam" is:** every run trains on 80 factory lessons, then is
quizzed on 8 lessons it NEVER saw (held-out). A run only counts if it
passes on unseen lessons — that is generalization, not memorization.

## Decision rule for the Phase-1 checkpoint

Pass the token exam AND the conversation gate:

1. **Token exam (probe):** timing tokens fire on unseen lessons (t+30s
   ideally; t+5s acceptable for the first gate) with `<sil>` ≥97% and
   over-firing bounded (thief steals ≲5 per lesson, vs v3's ~150).
2. **Conversation gate (T1–T5 live harness):** the checkpoint drives
   DuplexThinker through the scripted conversations; energy metrics must
   match the THRESH bar (reply after barge ≤800 ms, response p50 ≤1500 ms,
   no yielding on backchannels, ≤2.0 s false engagement).

## Dose-response series (all: 400 steps, frozen brain, 10 new-token rows only)

| run | jobs (train→exam) | recipe (data × weight) | t+30s | t+5s | `<sil>` | `<yield>` | words | verdict |
|-----|-------------------|------------------------|-------|------|---------|-----------|-------|---------|
| v1 | 6728455 → 6742556 | old data, 32 steps | 0% | — | 95–98% ✓ | 0% | ok | Smoke test: pipeline works end-to-end; timing data missing |
| v2 | 6742755 → 6745751 | 80 balanced shards ×1 | 0% prob 0.000 | — | 99.4–99.7% ✓ | 0% | 20–48% | Baseline: rare tokens (0.06% of positions) never win plain scoring |
| v4 | — → 6751164 | 80 shards × **w10** | 0% prob 0.15 | — | 97–100% ✓ | 0% | ok | w10 warms timing, safe — but not enough |
| v3 | 6746779 → (+6756308 thief) | 80 shards × **w50** | **50% ✓** prob 0.55 | — | 53–98% ❌ | untested | halved | 50× TEACHES timing; the cost is precision (thief below) |
| v5 | 6749313 → 6751180 | **5× data** ×1 (shards-v5) | 0% prob 0.000 | — | 97–100% ✓ | PARTIAL (1/6 at 100%, rest 0.03–0.76) | ok | `<yield>` responds to data spread, timing does not |
| v6 | — → 6754413 | 5× data × w10 +yield (shards-v6) | 0% prob **0.000 exactly** | — | 97–100% ✓ | weaker than v5 | best (0.10–0.39) | DATA×WEIGHT "law" is DEAD — adding data at fixed weight killed the w10 signal |
| v7 | 6756309 → 6756310 | 80 shards × **w30** | 0% prob 0.014–0.032 ❌ | **learned**: 1/6 at 100% (prob 0.772), rest 0.047–0.372 | 95.0–99.8% ✓ (mild thief: 4–5 steals/lesson) | 0% | ok | t+5s crosses between w10–w30; t+30s needs ~w50; non-monotonic: t+30s DROPPED vs w10 (frequent t+5s grabs capacity first) |
| v8 | 6756384 → 6756385 | 80 shards × **class-cost 50** | 0% prob 0.000 | 0% prob 0.000 | 99.3–100% but prob **flat 0.250** everywhere | prob 0.250, acc 0 | ~0 | **FAILED — math bug (below). Loss was label-blind; model only learned "never say rare words"** |
| **v9** | 6758553 → 6758554 | 80 shards × **w50 + thief-penalty 10** | 0% prob 0.000 | 0% prob 0.000–0.001 | **99.9–100% ✓ prob 0.958–0.995 (best ever)** | 0% | 0.00–0.13 ❌ | **OVER-SUPPRESSED — "the Monk": perfect silence, everything else dead. The tax fires at every non-rare spot (~1500/lesson) vs the up-push's ~1, so pen 10 swamped w50. Bracket: pen 0 → timing 50% + ~150 steals/lesson; pen 10 → nothing. Next: pen 2** |

## The v8 post-mortem (why class-cost failed)

Intended: rare tokens cost extra BOTH when correct (up-push) AND when
wrongly predicted (down-push). Actual math:
`L = Σ_j c_j·p_j − c_y` ⇒ `∂L/∂z_k = p_k·(c_k − E[c])` — **no y
anywhere**. The "when correct" half (−c_y) is a constant: zero gradient.
So the only learnable content was "suppress rare tokens everywhere",
which is what the exam showed: all timing 0.000, and sil/yield parked at
a context-free 0.250 in every scenario (a learned constant, not a
decision).

**v9 = the corrected idea, as two working pieces:** the 50× up-push
stays in the label-dependent weighted scoring (v3), and the down-push is
a separate `--thief-penalty` that taxes rare-token probability mass ONLY
at positions whose correct answer is not a rare token — exactly the
thief's turf (584/588 sil spots stolen by t+30s in v3).

## The thief (named 6756308, v3 failure-argmax histogram)

v3's "broken silence" was never broken silence — it was `<t+30s>`
OVER-FIRING at silence positions (scenario 029: 584/588 sil failures
were t+30s steals; 014/017/018 same). Timing recall fine; precision
broken. Per-position weights boost only the up-push; the down-push at
silence stays weight 1. This is what v9's tax fixes.

## Gotchas that cost us (do not re-buy)

- **probe-p1.slurm: always pass SHARDS_DIR** — without it the exam
  silently includes TRAINING rows (header prints "training set was 0
  shards"). First v4/v5 probe attempt (6749312/6749314) was discarded
  for exactly this.
- **Env by inheritance, NOT `--export`** — sbatch `--export` splits
  values on commas, silently truncating CLASSES=timed_task,barge_in
  (probe 6746780).
- **Submit from $DL/repo** — relative log paths follow the submission
  cwd.
- **Checkpoint provenance:** each `p1_checkpoint.pt` stores its `args`
  dict — the definitive record of what any run actually was (v2 was
  found hiding in `runs/p1-v1` this way).
- **This setup has CLIFFS, not slopes** — v6 (k5×w10) and v9 (pen 10)
  both landed timing at prob 0.000 EXACTLY, and v9's words also died.
  Dials near a cliff need bracketing (bisect between the last working
  and first dead dose), not extrapolation.

Checkpoints preserved: `runs/p1-v1-32step`, `runs/p1-v2-400step-unweighted`
(renamed from p1-v1), `runs/p1-v3` … `runs/p1-v8`.
