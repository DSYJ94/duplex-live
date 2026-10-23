# Native reasoning teacher lifecycle preparation — 23 October2026

The opt-in `evals/native_reasoning_lifecycle.py` boundary is implemented and has
eight passing offline lifecycle tests. The separate opt-in
`evals/native_reasoning_backend.py` transport has three passing offline tests:
pending HTTP does not block cancellation, late stale bodies are retained, HTTP
failure never retries, and existing output cannot be overwritten. It records
exact request/response hashes, provider/client request IDs and actual HTTP clocks;
credentials stay local and unrecorded. No actual backend call was made in these
tests. Neither component is connected to the current GPU pilot, production voice
client or a teacher collector. No teacher or model is approved by these tests.
The human approved the exact40 synthetic utterances; their complete source-v3
word/waveform audit now passes. The one-time20USD top-up remains pending.

A native voice delegation event must supply each distinct task identity. The
boundary accepts only actual chronological USER transcript deltas that have
already arrived; unknown annotation fields and future words reject. It copies
and hashes that prefix, preserving the complete earlier problem when a later
budget correction arrives. No case family, oracle, expected answer, forced
request, text-based difficulty rule or scripted cancellation enters its payload.
The model backend chooses one answer/cancel/replace action. This corresponds to
an external backend `answer` route after the voice model has requested help;
it does not treat a private-record lookup as external reasoning.

The HTTP call must run outside the ledger lock. Every complete provider response
is preserved with its response ID/hash, backend model, exact request/prefix hashes
and distinct actual dispatch/completion clocks. A replayed provider ID cannot be
rebound to another task. Cancellation/replacement validates the entire selected
pending target set before changing any target. It retires result delivery only;
it never claims an executing remote model was aborted. Replacement advances the
logical revision and keeps unrelated work. An old result that finishes after
retirement remains in raw evidence and cannot be delivered or mutate other tasks.

A transport delay is declared and recorded per request BEFORE backend dispatch;
a slow original must not silently impose that same artificial delay on a later
control reply. It cannot be adjusted after seeing a result. Each declared delay
has its own eligibility clock. Provider completion
is never relabeled as a slower remote computation. A bounded asynchronous wire
write serializes with cancellation commitment. Failed/uncertain writes never
retry, avoiding duplicate commentary. Successful writes record local completion;
they do not assert server receipt, native context-span application, audible
playout or physical hearing. Those need actual frontend/context/span/audio traces.

For eventual source replay, follow-up speech must be triggered after an actual
native request and backend dispatch, rather than at an invented dispatch time.
The complete32 source proposal's delivery delays are minimums. Declare any larger
transport-delay schedule before capture, based only on independently measured
source clip duration and a fixed response window, never on whether the teacher
has produced a desired answer. For example, a declared floor may be the greater
of the authored minimum and0.56seconds plus follow-up user WAV duration plus a
fixed4second response window. If the teacher misses that window, preserve and
fail the episode; do not wait for a correct answer before releasing the result.
Actual backend computation/delivery clocks and full native1100-frame episode
limits still apply. No timing schedule or source plan was changed by this note.

Still required before collecting an eligible teacher corpus:

- Source collection and whole40-object/32-scene/56-clip word/waveform
  qualification are complete; bind that exact version into the future collector.
- Wire this opt-in boundary to an actual GPT-Live teacher capture with continuous
  real input pacing and all raw output packets retained, without output filtering.
- Wire the bounded backend transport and request-ID/input-hash/response association
  to actual native requests; the lifecycle itself makes zero API calls. Offline
  fixture classifications cannot qualify a teacher. No direct/basic request is
  forced; unnecessary delegation must fail.
- Measured dispatch-dependent source replay, draining/finalization, immutable raw
  provider/WAV/event receipts, complete32 independent words/audio/semantic review,
  actual-time conversion and complete native numerical/gradient/retention approval.
- A distinct bounded training pilot, independent optimizer/RNG/native restoration,
  full native behavior/independent speech and sealed fresh qualification before
  adoption. Existing28 histories/49 teachers/full154 objectives and losses remain.

The current control40 pipeline, immutableC9 source, gates, data and one-eight-update
limit remain unchanged. This preparation neither extends nor adopts a candidate.
