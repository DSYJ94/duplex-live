# Native behavior coverage: what paired20/speech20 establishes

Reviewed 24 October 2026 on `main`, source `05352d75e80d162f905114c048e2ce2daecec4d6`.
Research and static evidence review only. No sealed fresh case payloads or owner
conversations were inspected; no model, recognizer, provider, GPU, SSH or
infrastructure work was run. The live GPU job was untouched.

**Finding:** the complete paired20/speech20 is a necessary regression and
measurement-integrity check with useful examples of four production objectives.
It is not complete production coverage, and currently recorded results fail the
covered behaviors. Context-sensitive backchanneling has no positive test in this
plan. This note proposes additions; it changes no gate or approval.

## Coverage present and missing

`paired20` means the **same ten development cases on two models**. `speech20`
independently transcribes those twenty outputs; it adds no new conversational
scenarios. The plan explicitly reuses TRAIN/development audio and excludes fresh
qualification. [Plan, lines 4–31](../../evals/fixtures/native-v2-development-20261019.json),
[paired runner, lines 137–201](../../portable/context/native_development.py),
[speech follow-up, lines 75–112](../../portable/context/native_speech_followup.py).

| Production objective | What the ten cases exercise | Coverage gap / inference limit |
| --- | --- | --- |
| Ordinary reasoning | Cases 0–3 require correct explanations of floating ice, sound, photosynthesis and cold-feeling metal, with independent speech agreement. | Four familiar synthetic science anchors cannot estimate broad reasoning or conversational accuracy. Backend execution is disabled in these six observe-only cases (0–5): retrieval markers are counted but not executed, so unnecessary delegation is not tested end to end. |
| Explicit cue-word silence | Case 4 requires silence until “ready,” followed by the timer answer. | One cue/task trajectory does not establish robustness to quoted cues, distractors, cue replacement, long delays, new voices or noisy audio. An audible “uh-huh” during explicit silence remains a failure. |
| Pause/resume | Case 5 requires book/shelf once, silence during pause, then cupboard after “go,” without restarting. | One scripted sequence does not cover arbitrary interrupted reasoning, repeated pauses, changed instructions or distinguishing user hesitation from yielding the floor. |
| Asynchronous delegation while answering basic questions | Cases 6–7 use the same user audio with changed private lookup values; case 8 adds missing-project clarification; case 9 requests actual cancellation. Correct receipts, result dependence and unrelated ordinary answers are intended outcomes. | These are synthetic record lookups, not matched easy-direct/hard-delegate reasoning decisions. The runner sets no minimum lookup delay, so actual event clocks must establish whether the unrelated answer really occurred while work was pending. Long delays, timeout/error recovery and result replacement are not a systematic matrix here. |
| Context-sensitive “umm/uh-huh” | No case expressly rewards an appropriate assistant backchannel or tests a benign user backchannel against a genuine interruption. | Silence compliance cannot establish natural short feedback. Neither can repeated generic acknowledgments. User hesitation, user assent and assistant feedback need separately labeled expected behavior and timing. |

Task ownership and the above expected outcomes are explicit in the
[unsealed plan](../../evals/fixtures/native-v2-development-20261019.json), with
the existing [prompt-by-prompt explanation](../NATIVE-MODEL-FAILURE-DIAGNOSIS-2026-10-23.md).
Observe-only limitations are stated in
[native execution, lines 16–20 and 60–76](../../evals/context_spanning_native.py).
The [command, lines 82–100](../../portable/context/native_development.py) omits
the optional delivery delay, whose [default is zero, lines 63–64](../../evals/context_spanning_probe.py).
Real-ASR mode records native request markers, actual worker returns, injection
time, input backlog and causal simulated playout; its report expressly leaves
microphone acceptance and speech quality unqualified.
[Execution, lines 110–219 and 252–273](../../evals/context_spanning_native.py).

## Completion, passing and measured outcomes are different

The fresh-entry validator binds exact checkpoint/runtime/artifact identities,
byte-identical baseline/candidate user PCM, parity, unforced execution and all ten
independent speech records. It requires candidate ordinary4 correctness, `pass`
verdicts and WER ≤0.08 with valid word clocks for those four. For cases 4–9 it
requires complete reasoned review but explicitly accepts `pass`, `partial` or
`fail` verdicts. This is authorization to measure fresh outputs, not an all-task
pass or adoption. [Validator, lines 34–137](../../evals/native_control40_fresh_review.py).
Production/adoption requires separate evidence that every required behavior
passes; permission to collect fresh evidence must not be described as that pass.

The speech auditor measures agreement with **the model's own generated text**,
keeps silent failures and remaps timestamps to the original WAV; semantics and
human listening remain pending. An intelligibly spoken false explanation can
have WER zero. Nonlexical vocalizations also require waveform/listening review,
because transcript absence alone cannot certify silence or proper backchanneling.
[Auditor, lines 1–4, 14–45 and 139–177](../../evals/moshi_speech_audit.py),
[follow-up completion flag, lines 111–112](../../portable/context/native_speech_followup.py).

The committed 23 October default-runtime review records ordinary cores **2/4 for
native520 and 0/4 for native7590052**. Both fail cue silence, pause/resume and all
complete asynchronous cases; the candidate's changed-value case is only partial.
The archive-runtime comparison records **1/4 each** and fails cases 4–9 for both.
Neither review performed human listening or adopted a model/runtime. These are
saved measurements, not a fresh run on 24 October and not results for the live
repair job. [Default review](../NATIVE-PAIRED20-DEFAULT-RUNTIME-REVIEW-2026-10-23.json),
[archive review](../NATIVE-PAIRED20-ARCHIVE-RUNTIME-REVIEW-2026-10-23.json).

## What the Kyutai primary sources support

The June [official post](https://kyutai.org/blog/2026-06-10-interactivity/)
identifies the relevant work. Its [paper §§3–4](https://arxiv.org/html/2606.11167v1)
separates pause handling, turn-taking, backchannels and interruptions, and tests
static audio plus dynamic multi-turn conversation. Its pause reward penalizes
speech intervals **longer than one second**; its backchannel reward treats shorter
intervals as feedback and matches timing within ±1 second. **Inference:** those
metrics cannot replace our stricter explicit-silence requirement or establish
semantic appropriateness of every short vocalization. The paper adds a content
reward because timing optimization can degrade answers. The
[released PersonaPlex-RL card](https://huggingface.co/kyutai/personaplex-rl-seamless)
links official inference code, requires access acceptance and CC BY-NC plus NVIDIA
terms, and describes research use. It supplies a comparator, not our qualification.

The April [official post](https://kyutai.org/blog/2026-04-30-moshi-rag/),
[paper §§3–5, Table 1 and Appendix C](https://arxiv.org/html/2604.12928v2) and
[official code README](https://github.com/kyutai-labs/moshi-rag/tree/8c6dfc101b7871baa428424bcdc583b74fb561d9)
support asynchronous retrieval while conversation continues. Pre-retrieval speech
includes coarse answers/fillers; this does not demonstrate answering an unrelated
new question during arbitrarily long work. Reference correctness and spoken-answer
correctness are separate; external API experiments assume 1.5-second retrieval,
and accuracy declines at longer delays. The pinned
[RAGManager](https://github.com/kyutai-labs/moshi-rag/blob/8c6dfc101b7871baa428424bcdc583b74fb561d9/moshi/moshi/inference_utils/rag_manager.py#L52-L90)
cancels pending generation and returns empty references on timeout/error.
[Channel reference encoding](https://github.com/kyutai-labs/moshi-rag/blob/8c6dfc101b7871baa428424bcdc583b74fb561d9/moshi/moshi/inference_utils/channel.py#L146-L181)
launches a separate update task without a visible request-generation check there.
That static observation does not establish an execution race or demonstrate user
cancellation guarantees or stale-result protection. Those require our own measured
lifecycle tests.

The posts' article bodies were not exposed by the text browser in this recheck;
technical claims above were verified against the linked papers, official model
card and pinned code, rather than inferred from post titles. This preserves the
evaluation/training and oracle/live distinctions in the
[existing research map](../FULL-DUPLEX-RESEARCH-MAP-2026-10-21.md) and
[earlier Kyutai review](../KYUTAI-INTERACTIVITY-MOSHIRAG-REVIEW-2026-10-20.md).

## Proposed follow-up, not an adopted gate change

1. Preserve complete paired20/speech20, ordinary4, every existing receipt/quiet/
   pause/causality check, restore and sealed-fresh requirements. Report execution
   completion, all-case review, per-behavior success and adoption separately.
2. Predeclare a separate unseen behavior matrix: varied ordinary explanations;
   exact silence with cue distractors/replacement; repeated interruption/resume;
   matched easy-direct/hard-delegate/clarify tasks; measured fast/slow/failed
   requests; unrelated basic answers while pending; cancel/replace/late results.
3. Add matched natural-conversation cases for user “umm” hesitation, user “uh-huh”
   assent, a real stop request, and appropriate assistant feedback. Score content,
   onset/duration, false takeover, missed feedback and subsequent continuation.
   Include explicit-silence counterparts in which every audible acknowledgment
   fails. Pair waveform timing with independent words and human listening.
4. Separate correct delayed oracle-result delivery from real ASR/backend/tool
   performance. Record request identity, pending interval, result availability,
   first informative spoken word, cancelled/stale delivery, backlog and audio
   continuity. Introduce additions through a versioned review; do not alter the
   running experiment or train on evaluation outputs.
