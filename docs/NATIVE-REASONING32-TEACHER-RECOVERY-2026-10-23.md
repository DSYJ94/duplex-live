# Whole32 teacher recovery —23 October2026

This is a new opt-in recovery; the failed v1 collector, stream, backend/lifecycle
code and all six raw attempts stay unchanged. Source-v3 still contains the exact
same independently qualified40 user recordings/56 clips/32 TRAIN scenes.

`evals/native_reasoning_teacher_v2.py` binds the tested completion/ack-aware
stream_v2 to actual capture and independent raw readback. The recovery module
requires the exact failed parent collection SHA34278d0e3014613a86d5066ded971055343143fc533809824210bece294f3c6b
and the independently backed-up88-file inventory SHA7e21d018b38cdc40cf2ac177aa3b99a1e41b0a64b7635fd52b71f068096cc8b2.
It rehashes all parent bytes and rechecks actual transport and lifecycle evidence
before making any new call.

The entire parent is copied inside the new recovery root. Four original records
000,001,004,005 are copied byte-identically, with explicit v1 lineage and original
receipt hashes. Their words/audio/spoken semantics remain unapproved. Nothing is
selected as an adopted passing subset.

Capture order is fixed before new model calls:

1. Repair only diagnosed002 cancellation and003 replacement using stream_v2,
   at most two simultaneous actual sessions. Stop new work if either fails.
2. Capture previously unattempted013 first. Its45.75-second question gives only
   five reference frames(0.4seconds) of margin under1100, before unknown native,
   provider, acknowledgment or playout latency. This is a planning risk, not proof
   of fit. Full measured prefix/live/all-sent-span accounting must pass.
3. Only then capture the other25 previously unattempted scenes, at most two
   simultaneous sessions. Stop new submissions after any transport, frame or
   lifecycle/oracle diagnostic failure; preserve already active attempts.

At most28 new voice sessions,34 aggregate attempts including the original six.
Each episode retains the six-native-request cap: the new maximum168 backend
requests plus five prior calls is173, inside the declared192 aggregate ceiling.
No automatic retry or resume into an existing output directory. The same local
collection lease prevents duplicate collectors. API keys stay local; source
WAVs, words, voices, quality bars, folds, native prefix and old retained training
objectives are unchanged.

Voice and backend payloads use the same general interface for every case. The
voice model must create requests itself; no case-family classifier, forced RET,
expected answer or oracle enters its input. An explicit general instruction
asks the voice model to request an actual application-state update for pending
cancellation/replacement, while answering basic arithmetic itself. Context uses
only actual arrived USER transcripts. Source scoring annotations are read only
after capture or for the isolated reference-packing diagnostic, never for model
routing, returned content, output suppression or answer-dependent release timing.

After all32, the mixed-version verifier reconstructs each original/v2 episode
with its actual recorder, verifies all original/provider bytes and full frame
accounting, and checks lineage. Whole32 independent pinned-ASR words/audio,
independent spoken semantics, actual-time conversion/native assembly, numerical
and gradient/retention approval remain required. The planning estimate and
lifecycle diagnostic cannot approve teachers. Never crop or weaken a limit to
make013 fit; a genuine measured packing failure requires its own diagnosis.

This recovery does not modify the running C9 control40 training pipeline, purchase
compute, extend a failed model, change an optimizer, unfreeze the base or adopt
a candidate. At implementation time no new recovery teacher call had been made.
