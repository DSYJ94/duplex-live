# Native silence control: diagnosed failure and next work

Candidate7590052 is not ready for adoption or further unchanged extension.
The owner reported speech and keyword questions after a silence-until-cue request.
The server had no runtime exceptions; owner audio/transcripts were not recorded.
The nine complete locally synthesized diagnostics reproduce speech during the quiet
interval across Ready/Geronimo, no cue, and a delayed cue. All transport requests
passed; no behavioral quiet constraint passed. Exact source hashes, generated text,
frame evidence and waveform hashes are in the adjacent diagnostic JSON. This is
stochastic development evidence, not independent speech/full-task qualification.

The existing approved TRAIN014 Ready user stem separately stayed near silent
inside6–9.5s; first words followed the actual second utterance. Its later answer
was repetitive. Therefore do not infer that the candidate has no silence skill at
all, or that the novel cue alone explains failure. Wording/voice/state generalization
are unresolved. The broad project-clarification prefix is suspect, but scoping it
alone did not fix the three cases. Appending a generic silence rule also failed.
These per-session prompt diagnostics supplied no cue value or output words, changed
no model/precision/clock, and left the live audition default unchanged.

## Bounded next proposal — prepared, not submitted

1. Run the complete original baseline/candidate native10 and independent speech
   review with the retained ordinary4/4 requirement. Restore exact state separately.
   Do not call these nine diagnostics a passing subset or a replacement for that gate.
2. Audit existing TRAIN quiet/pause targets and masks against actual audio and
   approved native captures. Identify cue diversity, silent-frame coverage and
   exposure under the actual154-objective recipe before changing a loss or dataset.
3. Prepare a NEW versioned synthetic TRAIN control expansion with diverse unseen
   cue words, paraphrases/voices, long waits, distractors, near-miss cues, and a
   later valid cue followed by useful task completion. Acknowledgments, keyword
   clarification and fabricated content while quiet are explicit failures. Keep
   the current Geronimo diagnostics DEVELOPMENT, never relabel them fresh holdout.
4. Complete teacher→independent quality→semantic→conversion→native numerical
   approvals for the ENTIRE new TRAIN collection. Keep originals, exact words,
   audio hashes, receipts and folds. Never edit an approved clip to conceal a
   student failure. Existing24 and old25 teacher approvals remain unchanged.
5. Only after those approvals, prepare one explicitly bounded fresh repair pilot
   from a reviewed baseline, preserving all28 actual histories, KL/hidden retention,
   all49 approved histories and held-out hashes. Any changed recipe is a new stage,
   not an exact-resume claim. Do not extend this failed candidate unchanged or
   weaken the existing extension/adoption guard. First establish complete native
   behavior, then all40 sealed paired fresh outputs before adoption.

No new training, data approval, output muting, model adoption, package change,
owner recording, or additional purchase has been performed by this proposal.
The live trial remains budget-guarded through21Oct17:28:42UTC. A complete external
worker evaluation needs local authorization in the ignored0600keyfile and two
actual distinct GPUs under a concrete budget. Neither is simulated by these probes.

## Initial source audit completed

The actual741872f native-stream loss already assigns half its normalized mass to
spoken targets and half to clock/control targets, with PAD3 weighted.5 inside the
clock group. TRAIN014 has417 active targets and7 spoken targets, and its native
auxiliary was exposed once in the eight-update stage. Silence was not omitted.
The run deliberately detaches autograd at each recorded native call while keeping
forward cache values. Generalization/coverage, limited exposure and long-range
credit assignment are hypotheses, not established causes. Diagnose these before
assuming more unchanged updates or adding a duplicate silence loss will fix it.
Exact source/report fingerprints and actual TRAIN014 row:
[NATIVE-QUIET-SUPERVISION-AUDIT-2026-10-21.json](NATIVE-QUIET-SUPERVISION-AUDIT-2026-10-21.json).
