# duplex-live — final report (2026-10-12)

**Plain words first.** The goal was a live duplex voice model: it hears
you, it knows when to speak, it speaks. We got most of the way and
proved exactly where the remaining wall is. The mic demo runs today.

## What was built and proven

1. **A model that hears in time.** The hardest bug of the project was
   not in training at all: the model's ear produced 52 sound-snapshots
   per 4 seconds while its clock ticked 50 times, and the surplus was
   silently discarded — everything it heard drifted 4% late. One fix
   (resampling the ear onto the tick grid) took turn-taking recall
   from 0.14 to 0.93 overnight, same model, same data, same recipe.
2. **Turn-taking that survives contact with strangers.** On
   conversations the model never trained on: it starts 60–95% of the
   turns it should, never once talks over the user, and places word
   starts within a second (most exactly). A small trained gate picks
   the speaking/quiet operating point.
3. **A live 80 ms loop.** Mic in → ear → decision → voice, one tick at
   a time. Decisions take 59 ms. The mic demo (`demo/mic-demo.sh`)
   runs end to end over SSH to the cluster.
4. **A data factory that cannot lie.** Every conversation is verified
   clip-by-clip (speech-to-text vs script, repair-in-render) and
   word-by-word (timings measured from audio, not estimated). Training
   jobs refuse to start on unverified data. 515 verified conversations
   across open TTS, ElevenLabs, and GPT-5-authored banks.
5. **A method that closed wrong paths fast.** Three families of fixes
   were ruled out with evidence rather than opinion: readout heads on
   trunk states (twice, at chance even after the ear fix), decode-side
   repairs (all arms worse), and exogenous gating (lands on the same
   frontier).

## Where it stopped, honestly

**The model cannot generate its own words in free-run.** Trained
purely on correct scripts, it places words perfectly but produces
gibberish when self-driving (the "digit soup"). The final experiment —
434 conversations, self-history training — did not fix it and muted
the dialog class. Conclusion with three-way evidence: word content
needs the next architecture phase, not another round on this trunk.

**Delegation behaviors: learned, but not yet mixable.** The model
fired `<delegate>` at exactly the right moment in its own lesson — the
skill exists. But mixing 81 delegation lessons into training broke
dense turn-taking (class interference: 146 false delegates on one
lesson). The balanced-dilution attempt (in flight at writing) tests
the cheap fix; either way the finding is documented.

**The serving wall.** Word-dense turns cost ~5× the 80 ms budget on
one GPU. Published fixes exist (speculative decoding for this exact
model; a 160 ms decision grid is the sanctioned fallback). Not a
research problem.

## The next phase, in one paragraph

Wire the Qwen3-Omni Talker (the speech-generation head already in the
base model) into the duplex loop, train on a mixture balanced across
behavior classes, add real two-channel audio for interruption
statistics (DuplexChat, 282k h, open), and apply RL for timing polish
after supervised content exists. The token vocabulary, grid, loop,
data pipeline, and evaluation instruments all carry over as-is.

## The artifacts

- Mic demo: `demo/mic-demo.sh` (turn-taking live; words from the
  script bank — the honest claim)
- Gallery: `python -m viz.listen` (timing-demo, live-demo batches)
- Code: fail-closed factory (`data/synthetic/`), training
  (`layera/`), live loop (`m2/`), all committed with full history
- Records: `EXPERIMENTS.md`, `HANDOFF.md` (every verdict, dated),
  `docs/research/2026-10-08-duplex-approach-survey.md` (cited survey)
- Checkpoints: `runs/scale1-q2-pt1` (the delivered model),
  `runs/scale1-q2-gate1` (its gate)

**The fresh-set verdict (2026-10-12, one shot, as reserved all
project).** On barge-115/117/119 — never trained on, never evaluated
until now — the delivered model scored F1 0.80/0.86/0.80, recall
0.87/0.99/0.95, and word onsets 12/12 within one second (9/12
exact). Content WER ~1.9, the documented limit. Turn-taking
generalizes; that is the project's result.

*Allocation ends 2026-10-21. Remaining: balanced-delegation round
verdict (in flight), final gallery refresh, hand-back.*
