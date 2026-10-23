# Native interaction and delegation training coverage

The native Moshi/PersonaPlex duplex streams and Context Spanning result interface
already exist. Additional training targets when to listen, acknowledge, take or
yield the floor, answer directly, or request external reasoning. It must retain
the complete old49 teachers/all28 actual histories/full154 objectives/KL1+hidden10.
The current control40 repair remains a required whole-corpus gate; this plan does
not approve its failed batch or create a passing subset.

| Behavior | Positive supervision | Matched negative/control |
| --- | --- | --- |
| Listen through an unfinished turn | Preserve the user's unfinished thought across short/long internal pauses | Do not infer turn completion merely from a gap |
| Take the floor | Answer once when the user has yielded; correct content and measured onset | Avoid permanent silence or premature completion |
| Listener backchannel | Brief contextual acknowledgment during user speech without taking over | Remain completely silent under explicit quiet, sensitive timing or interruption |
| Hold own floor | Occasional natural hesitation if useful; continue the same answer | Avoid repetitive filler, fabricated work claims or filling every backend wait |
| Interruption and resume | Stop, retain unfinished task, resume only when permitted | Do not restart completed steps or mistake the user's backchannel for cancellation |
| Direct versus delegate | Matched easy/hard tasks; native direct answer for basic cases, complete scoped request for hard/tool cases | Penalize unnecessary requests and unsupported direct answers separately |
| Delayed result integration | Keep listening, preserve request context, use the actual returned result | No answer before needed evidence; no fabricated completion |
| Cancellation and changed intent | Respect cancellation/replacement and respond to the current intent | Reject stale results, unrelated results, duplicate deliveries and error-as-success |

These are proposed additional families, not generated/approved examples. Source
scripts/audio, native teacher histories, receipts, independent recognition/quality,
semantic review, measured conversion and native numerical/gradient scope need the
same explicit provenance as current data. Keep held-out audio, voices and folds
separate. No external dataset/checkpoint import, new base, new optimizer or new
API/GPU job is performed by this document.

Backchannels such as “uh-huh” are listener behavior; “umm” can be speaker hesitation.
Do not insert either periodically or optimize their count alone. New nonlexical
vocalization data needs an explicit audio/annotation protocol; it cannot silently
be accepted by spelling substitutions or a weakened lexical ASR rule. Word-based
acknowledgments still need exact independent word/timing checks. No existing
control40 threshold or quiet interval changes.

Evaluate each axis separately on fresh complete conversations: false takeovers,
missed valid replies, backchannel precision/timing, explicit-quiet violations,
interruption latency and semantic continuation, direct-answer accuracy, delegation
need/argument correctness, result fidelity, cancelled/stale delivery, failure
handling and end-to-end latency. Good backend text alone is insufficient. Include
oracle-result and real-backend runs to distinguish routing/integration from worker
failure. The runtime must also enforce request identity, cancellation and result
ownership; model training alone does not supply those guarantees.

Kyutai's primary study separately trains pause handling, turn-taking, backchanneling
and interruption, retaining a response-quality objective. It supports these
categories, not a claim our small LoRA already achieves them. Its broad RL recipe
is a separate larger workflow; do not replace the present supervised repair with
an unbudgeted reproduction.
[Paper](https://arxiv.org/html/2606.11167v1).


## Authored reasoning/delegation source proposal — 23 October

A distinct, deterministic [32-scene TRAIN source proposal](../evals/fixtures/native-reasoning-delegation-source-proposal-20261023.json)
and [generator](../data/synthetic/native_reasoning_delegation.py) now cover eight
matched public reasoning problems in each of four conditions: direct basic
arithmetic, background constrained reasoning while answering a basic question,
cancellation, and replacement after a changed budget. This includes reasoning
without private-record lookup. The native model's request decision is separate
from the external reasoner's `answer` route.

Each hard task selects exactly three of six items under a budget, a required item,
an incompatible pair and explicit tie rules. An exhaustive combinations solver
produces annotation targets; an independent64-bitmask enumeration checked all32
required original/replacement solutions. Every replacement changes the selected
items. The candidate speech model and reasoner must receive only the actual user
words; annotations and oracle solutions are never inference inputs or fabricated
worker receipts. Cancellation/replacement must follow actual dispatch while the
original task is pending, with execution and delivery-delay clocks separate.

The proposal has32 cases and56 authored user utterances (2,845 words, maximum117
user words per episode). No audio, API calls, native teacher collection, numerical
approval or training has occurred. Existing TRAIN/validation/test folds and sealed
holdouts were not read or changed. The complete measured episode plus prefix/spans
must fit the existing1,100-frame bound without cropping. A new real asynchronous
reasoning teacher protocol and the whole32 independent quality/semantic/conversion/
numerical chain remain prerequisites; the historical private-record renderer must
not silently consume this new schema. Three source/solver/mutation tests pass.

This is the reasoning/lifecycle portion of the wider curriculum. Microturn,
backchannel and nonlexical vocalization data still need their own approved source
and audio annotation protocol; this proposal does not establish those behaviors.


## Whole-source binding audit — 23 October

The unchanged32-case source fixture passes exact deterministic regeneration and
independent combinatorial target checks. A mutation audit exposed a validator gap:
valid optimizer oracles could coexist with changed spoken questions/answers,
revision/delivery labels or retention declarations. Twenty-one intentionally
corrupted variants were accepted by the old validator. The repaired validator
binds the complete authored JSON for this version; all five tests now pass,
including the existing independent bitmask solver check.

This found no corruption in the original fixture and establishes no cause for
existing model failures: this source has never been recorded or trained. It
prevents mismatched future reasoning labels. No API call, new recording, source
audio repair, GPU source change or optimizer update occurred in this audit. See
[the exact audit](NATIVE-REASONING32-SOURCE-BINDING-AUDIT-2026-10-23.json).


## Reasoning source-audio collection adapter — 23 October

The separate `evals/native_reasoning_source.py` collector extracts only authored
user words and voices from the exact32-case proposal. Forty unique speech requests
cover all56 scene utterances; the original hard question WAV is shared byte for
byte across each matched delegate/cancel/replace condition. No answer annotation
or oracle enters any provider payload. The collector preserves provider WAVs,
request IDs and payload/WAV/receipt hashes, with no automatic retry, gain change,
resampling, timeline, teacher, inference or optimizer path. Fixture transports
are marked and rejected by production verification.

A separate pinned Whisper-turbo CPU audit checks every unique original source
object and its complete32/56 membership at WER≤0.05 and peak≤0.995, without a
reference prompt. This new long-question source schema retains the whole native
episode1100-frame bound: an individual clip cannot exceed88seconds, and actual
prefix, spans, worker/result and lifecycle time must still fit together during
subsequent measured conversion. The historical control40/old49 clip limits are
unchanged. Source-only success never approves a teacher or model training. Eleven
local source/quality/independent-solver tests pass. No real collection had run at
this source commit; actual receipts must supply its eventual status.


The new source-word audit explicitly records the existing `norm_words` numeral
expansion (for example2/two), pins that implementation's hash, and still rejects
wrong numeric values. No comparator behavior or0.05/0.995 threshold changed.
Twelve source/quality/solver tests now pass; collection still has zero API calls.
The separate offline lifecycle component now has eight passing tests, including
predeclaring a separate transport-delay floor for each request before dispatch.
It is unconnected to actual source/teacher capture and cannot qualify a model.
