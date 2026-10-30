# Native duplex foundation and cost review

Reviewed 30 October 2026 against primary papers, model cards, the official Venus repository and our current implementation. Research only: this note starts no compute, training, downloads, purchases or external model calls. Published results are author claims, not qualification of our product.

## Decision

**Do not start foundation pretraining. Keep native duplex interaction plus an asynchronous reasoning backend as the preferred product architecture, but make continued investment in Venus conditional on one decisive released-base/runtime comparison.** Reusing a pretrained text transformer is technically sound; the released frontend's actual English speech, responsiveness and control quality are the unresolved questions. This recommendation is an engineering inference from the evidence below, not a demonstrated GPT-live equivalent.

## Why Qwen ancestry does not disqualify Venus

The released approximately 9B Venus-Audio combines a Qwen3-8B language backbone, Whisper-Medium audio encoder and S3 speech-token/streaming flow-matching speech generation. Its parent is MiniCPM-o 4.5, rather than plain Qwen or PersonaPlex. These are distinct pretrained components integrated into a streaming model. [Venus model card](https://huggingface.co/inclusionAI/Realtime-Venus), [MiniCPM-o 4.5 card](https://huggingface.co/openbmb/MiniCPM-o-4_5)

Venus interleaves incoming audio with generated output in one-second units. The model predicts listen/speak/turn-end controls from its semantic state, including while a response is underway. Its post-training covers control, response and delegation spans; only the Thinker is updated, with the acoustic decoder fixed. The audio subset contains 865k samples, including 100k duplex and 90k delegation examples. Its private requests require host suppression from visible text **and speech**, causally captured evidence, and prepared backend replies. These details make its control native while leaving essential execution responsibilities in the runtime. [Venus paper §§4–5](https://arxiv.org/html/2609.13814)

The important distinction is **initialization versus trained operating mode**. A text-only transformer does not become duplex because a microphone is attached. It can become the semantic core of a native duplex system after audio alignment, causal streaming design and interaction training. Conversely, a model advertised as native duplex can still have poor reasoning, speech or timing. Architecture-level concurrency and product-level quality need separate evidence.

Moshi provides a useful counterexample to the ancestry objection: it initializes its Temporal Transformer from the text-only Helium model, adds audio modeling, then learns multi-stream interaction in subsequent stages. Its stream operates at 12.5Hz. PersonaPlex extends the Moshi/Moshiko family rather than Qwen. Thus neither lineage offers a principle that a native duplex model must have been multimodal from its first pretrained token. [Moshi paper §§3–4](https://arxiv.org/html/2410.00037), [PersonaPlex card](https://huggingface.co/nvidia/personaplex-7b-v1)

## What the evidence says about the current choice

Venus was a defensible candidate because the released system already couples streaming interaction to background work. Its authors' browser/Harness release is a separate integration from our Audio wrapper; an upstream demo must not be treated as an exact reproduction of our checkpoint and runtime. [Official repository](https://github.com/inclusionAI/Realtime-Venus)

Our current 406-update candidate is **not accepted**. The complete paired review uses 32 verified reused parent298 cases and 32 new candidate cases. Strict generated-text/audio agreement improves 10→12/32; correct requested lookups improve 0→2/4 in both delegation and pending-work families; genuine cancellation remains 0/4. Early voice falls 2→1/4 quiet cases, but only 2/4 meet the necessary no-early-voice-plus-post-cue-voice condition. These are fixed regression cases, not pristine generalization results. A correct lookup receipt alone does not establish a correct spoken answer. [Complete local review](NATIVE-VENUS-ACTION64-REVIEW-2026-10-30.json)

Current adaptation trains 851,968 last-layer q/v LoRA parameters using 216 TRAIN scenes and 52 DEV scenes; its acoustic decoder is frozen. It tests control and complete request fidelity, not a comprehensive speech-quality curriculum. Neither these data nor more updates to these weights establish a cure for voice naturalness. [Training results](NATIVE-VENUS-ACTION108-RESULTS-2026-10-30.json)

There is also a concrete attribution risk: our `NativeEngine` defaults `nonspoken_actions=False`, and our synthetic lookup worker accepts a narrower request contract than the published natural-language/evidence-aware Harness. Failures under that setup are failures of **our tested system**; they do not isolate Qwen intelligence or establish the released Venus system's ceiling. [Local engine](../portable/venus/native_app.py), [test collector](../training/venus/behavior.py)

## The cheaper route, and when to change it

| Route | Assessment for this project |
|---|---|
| Released native duplex frontend + strong asynchronous backend + targeted adaptation | Preferred. Reuses existing audio/language pretraining, focuses learning on interaction, task capture and faithful result delivery. Must retain basic frontend competence. |
| Different released duplex frontend | Appropriate if original Venus fails the essential voice/latency/control requirements under its intended runtime. Compare one credible challenger, not an open-ended model search. |
| Broader adapters or full-weight post-training | Conditional on a good base and evidence that current adaptation capacity/data is limiting. Changes a different problem from pretraining and needs preservation of speech/intelligence. |
| Foundation pretraining from scratch | Not justified now. We have no evidence that reusing pretrained components is the causal failure, and no demonstrated data/serving recipe that warrants replacing them. |

This split is supported by two independent directions. Thinking Machines describes an intelligent interaction model with asynchronous sustained reasoning, using 200ms micro-turns; it also trained its own interaction model from scratch. That demonstrates one design choice, not a requirement that our product repeat its pretraining investment. [Official system overview](https://thinkingmachines.ai/blog/interaction-models/)

MoshiRAG releases a native frontend with a separately replaceable retrieval backend and trained reference-injection path. It supports the division of labour, but its interface is not interchangeable with Context Spanning or Venus, and retrieval is narrower than our cancellation/pause/revision product goal. [Official MoshiRAG implementation](https://github.com/kyutai-labs/moshi-rag), [paper](https://arxiv.org/html/2604.12928v2)

Kyutai's interactivity work supports trajectory-level optimization after supervised learning: pause handling, turn-taking, backchannels and interruptions receive distinct rewards plus a semantic-quality term. Its reported 32-H100 GRPO recipe is not our cheapest next experiment. Released PersonaPlex-RL weights also carry noncommercial restrictions, so they are research comparators rather than an automatic production replacement. [Current paper](https://arxiv.org/html/2606.11167v2), [model conditions](https://huggingface.co/kyutai/personaplex-rl-seamless)

## One bounded decision before more training

1. Compare untouched Venus-Audio and saved406 on the **same complete existing 32 cases, English reference and decoding profile**. Keep text correctness, actual speech, control timing and backend receipts separate. This isolates adaptation effects under our host; shared failures do not yet isolate the foundation.
2. For shared failures, verify the implicated published runtime contract once: native audio perception, private-request speech exclusion, causal context capture and reply reintegration. Use the original released Audio inference path where available; clearly label any Omni substitution. Do not rerun an entire infrastructure qualification campaign.
3. Decide from those results. Good base/poor adapter means repair adaptation. Good text/poor audio means repair or replace the speech path. Persistent poor original English conversation or an incompatible reaction cadence means compare a different released duplex foundation. A one-second unit deserves particular scrutiny against the desired responsiveness; a smaller browser buffer does not change the model's learned clock. Unit size alone is not a proven universal lower bound on reply latency: onset, cutoff, buffering and computation need their own measured clocks.
4. Resume substantive training only on the selected viable foundation, with the diagnosed data/objective/capacity change and a fixed stopping/evaluation plan. Preserve all current artifacts. No further optimizer updates are authorized by this research note.

To approach the desired experience, measure user-perceived reply delay, meaningful-answer delay, interruption cutoff, long-session stability, spoken intelligibility/naturalness, correct direct answers and reliable background-work delivery. More parameters alone improve none of these by definition. OPAL can adapt a recipe; it cannot compensate for an incompatible runtime, absent supervision or an inadequate speech generator without those becoming explicit experiment changes.

The older [pretraining decision](00-pretraining-decision.md) contains exploratory cost/time estimates and the absolute claim that audio pathways must always be trained from scratch. Neither should be treated as a verified current plan. Reusing a qualified released audio pathway is a valid cheaper option. No exact cost-to-parity estimate or claim about proprietary GPT-live internals is established here.
