# Competent native reasoning, calibrated delegation, and useful compute

7 November 2026. Decision and implementation plan, not a claim that the experiments below have run. Owner direction: the live model should answer straightforward questions itself, understand conversation and recognise its limits, then delegate harder work without losing the conversation. Natural British speech and reliable duplex control remain equally necessary.

## Recommendation

**Improve verified reasoning and routing data, with native rehearsal, before another capacity increase. Use Isambard for the main training programme, including a measured single-node multi-GPU branch. Reserve continued pretraining and structural expansion for demonstrated deficiencies of the foundation. Do not pretrain a new foundation from scratch.**

This does not mean indefinite small pilots. Finish the admitted new-data tranche, establish a fresh reasoning comparison, and run complete training schedules with checkpoint/DEV decisions. Research should change the next experiment, not postpone it. Parameter count, training loss and a good transcript are each insufficient evidence of product quality.

## What the latest evidence actually says

The 698,351,616-parameter broad LoRA completed parent526→646: 120 updates, two complete 240-TRAIN epochs and full64 DEV boundaries. Eight saved updates were restored; only112 remaining updates were executed. Final independent Adam/RNG/OPAL/native-logit restore passed, and the final8.51GB checkpoint is hash-verified on the Mac. This is a substantial completed learning experiment, not an unstarted pilot. [Training receipt](NATIVE-ISAMBARD-TRAINING646-COMPLETE-2026-11-07.json).

| Same current speech runtime, 32 cases per model | Untouched foundation | Candidate646 |
|---|---:|---:|
| Spoken words match intended words | 12/32 | 15/32 |
| Raw peak limit passes | 22/32 | 30/32 |
| Both speech checks pass | 9/32 | 15/32 |
| Genuine pending cancellation | 0/4 | 4/4 |
| Correct initial key in cancellation cases | 0/4 | 2/4 |
| Necessary silence before cue and voice afterwards | 0/4 | 4/4 |

All outputs were copied back and hashed. Both models remain unqualified. This comparison shows that adaptation helped some control behaviour and that speech defects also exist in the untouched foundation/runtime combination. It does not identify a unique acoustic cause. Different generated content prevents interpreting runtime timing differences as pure kernel speed. [Matched review](NATIVE-RELEASE-CURRENT32-COMPLETE-REVIEW-2026-11-07.json).

Against526,646 reduced teacher-forced CE across every family, but ordinary KL increased from0.558662 to0.722494. That is a retention warning, not proof that its reasoning universally worsened. The previously reported wrong funnel answer is **now correct in646 generated text**; its audio loses the ending. Do not keep diagnosing that old text error as current. Our existing ordinary questions are too easy and too few to establish general reasoning. [Complete comparison](NATIVE-POSTTRAIN646-COMPLETE64-REVIEW-2026-11-07.json).

The frozen native LLM also failed a separate semantic-review calibration,13/16, accepting one false factual statement and two replies before their required feedback. This is evidence that it cannot approve our teachers under that protocol; it is not a general reasoning benchmark. No96 teachers were approved by that attempt. A frozen public Qwen3-32B reviewer is separately prepared, with the same controls. It does not replace the voice model. [Reviewer decision](NATIVE-QWEN32-REVIEWER-NEXT-2026-11-07.md).

## Lessons from the requested reports

These are authors' findings in different models and workloads. The applications in the right column are our proposed inferences, not reproduced Venus results.

| Primary source | Useful finding and proposed application |
|---|---|
| [Aleph Alpha: hierarchical scaling](https://aleph-alpha.com/en/blog/scaling-pre-training-in-practice-a-hierarchical-approach/) | Profile and narrow configurations at small scale before increasing hardware; local batch, accumulation, checkpointing and sharding interact. For us: compare one practical single-GPU configuration with two GPUs while preserving global batch and objective, then use four if worthwhile. Their30B-A3B/large-B200 results are not our throughput estimates. |
| [Aleph Alpha: Kolibri technical report](https://aleph-alpha.com/downloads/tech-report.pdf), especially midtraining | Curated mixtures, controlled repetition and evaluation after downstream instruction tuning matter. For us: measure concept coverage and repetition, preserve rehearsal, and compare actual native behaviour after reasoning training. Kolibri's78.1B-total/3.46B-active model is distinct from the30B scaling-blog model. Its trillion-token stages do not establish a minimum useful adaptation budget for us. |
| [Microsoft: frontier pretraining](https://microsoft.ai/news/frontier-pre-training-at-microsoft-ai/), [technical PDF](https://microsoft.ai/pdf/Frontier-Pre-training-at-Microsoft-AI.pdf?v=2) | Data-repetition assumptions must match when extrapolating proxy runs; optimizer settings interact; wall time to quality matters alongside FLOPs. For us: hold exposure comparable, log effective LR and weight decay together, and select on useful progress per allocated GPU-hour. Do not transfer frontier-model scaling coefficients or massive infrastructure requirements to LoRA. |
| [Phi-4 report](https://arxiv.org/abs/2412.08905), [full text](https://arxiv.org/html/2412.08905v1) | A14B model gained strong capabilities through curated real/synthetic data and training stages. Its mixture used40% synthetic data; its midtraining also addressed context extension. For us: validated problems, diverse explanations and short conversational answers are more relevant than copying its exact proportions, context curriculum or learning rates. Its text benchmarks do not demonstrate full-duplex speech quality. |

The common lesson is **choose data and optimization for the capability being learned**. None establishes that we must build a200B voice front end. Nor do they establish that our existing foundation is sufficient without testing it.

## Separate the four capabilities

1. **Native competence:** understand context, perform short arithmetic/causal reasoning, explain familiar technical concepts, recognise missing information and answer concisely.
2. **Allocation judgment:** choose direct answer, clarification or delegation. Difficulty is relative to this model and its tools, not simply prompt length or a keyword such as “technical.”
3. **Task transfer and result use:** carry the actual question, constraints, relevant conversational context and task identity into the backend; distinguish pending, completed, cancelled and stale results; explain the returned answer faithfully.
4. **Native communication:** retain listening, silence, interruption and natural British speech throughout. A correct backend answer that is spoken as gibberish is still a failed product interaction.

Delegation cannot rely solely on the model saying it is confident. Train and calibrate the routing decision against measured native/backend success. A small independent router or deterministic escalation rules can provide an interim backstop; they must not be reported as learned native routing. [RouteLLM](https://arxiv.org/abs/2406.18665) supports learning quality/cost trade-offs from preference data, but does not supply our asynchronous voice/cancellation protocol.

## First decision-changing evaluation

Prepare a fresh, versioned128-case reasoning cohort before selecting another candidate:32 everyday/math/causal,24 technical concepts,24 context and short multi-step problems,16 ambiguity/insufficient-information,24 harder delegation tasks, and8 unavailable-tool or stale/contradictory-result cases. Use disjoint problem/template families, entities and audio from TRAIN and the inspected regression cohort. Separate48 development cases may support adaptation; the128 final cases remain unavailable to the optimizer and failure-mining queue until the comparison is frozen.

Run the same cases through untouched Venus,646 and the next trained candidate. Obtain an independent stronger-backend reference where executable answers or source-grounded rubrics are insufficient. Compare four stages separately:

- Written question → generated answer: language/reasoning capability without ASR or speech rendering.
- Spoken question → generated answer: the additional effect of audio understanding/context.
- Native routing → actual worker/result: correct choice, complete arguments, execution and result grounding.
- Final waveform → intelligible correct answer: communication and meaning, not just word agreement.

Score task correctness, numeric/unit/constraint preservation, clarification appropriateness, unjustified certainty, missed and unnecessary delegation, backend completion, stale-result suppression, and end-to-end spoken correctness. Publish denominators and paired disagreements by family; do not hide hard cases in an average or accept a passing subset. Use executable oracles for arithmetic/code and blinded adjudication for explanations. A single LLM judge is not ground truth. Preserve source references and human disagreements.

This cohort is a screening and comparison instrument, not proof of frontier performance. Keep existing strict native/audio/retention gates unchanged. Before making a product claim, freeze product-level targets and assess a larger fresh cohort plus actual WebSocket/PCM conversation sessions. No reference answer enters inference prompts, no inspected HOLDOUT question becomes a TRAIN repair, and no owner conversations are recorded.

| Measured pattern | Next action |
|---|---|
| Foundation correct, adapted model wrong | Repair rehearsal, mixture, effective LR or retention weighting before adding capacity. |
| Both wrong on basic text questions | Teach missing concepts/short reasoning; compare a stronger reference. Consider selective continued pretraining if SFT does not transfer. |
| Text correct, audio-input answer wrong | Improve listening/context alignment and spoken training coverage. |
| Correct generated answer, wrong waveform | Acoustic conditioning/rendering work; more math SFT is not the direct fix. |
| Native weak, backend correct, routing weak | Train direct/clarify/delegate contrasts and complete context transfer. |
| TRAIN and DEV both stall after verified coverage and useful updates | Test recipe or capacity; distinguish underfitting from bad targets. |
| TRAIN improves but fresh DEV worsens | Reduce repetition, improve diversity/rehearsal and review regularization; do not blindly increase model size. |

## Data programme: useful sources and admission

| Source | Proposed use | Verified terms / limitations |
|---|---|---|
| Existing240 TRAIN plus new96 native scenes | Preserve control/history objectives; add48 basic/technical scenes and48 request/result/cancellation scenes. | Original folds/audio/receipts remain immutable. All182 new recordings pass the derived source checks; all96 measured timelines are compiled. Semantic review and native-cache approval remain pending. |
| [GSM8K](https://github.com/openai/grade-school-math) | Selected TRAIN arithmetic/word problems, then new independently generated family variants with executable numeric checks. | [MIT repository license](https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE). Never train on its test split; familiar benchmark scores may reflect foundation contamination. |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | Sample school-math, units, proportionality and short reasoning; validate answers independently. | Dataset card:14M examples,CC-BY4.0. Generated answers are not automatically correct. Deduplicate against GSM8K/MATH seeds and held-out families; do not turn14M long solutions into audio. |
| [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | Verified science/code/math/puzzle examples and executable tests, selecting short explanations suited to the front end. | Apache2.0. Long reasoning traces are not desired spoken output. Preserve provenance and check individual source constraints. |
| [Salesforce xLAM60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) | Schema/argument diversity; supplement with our direct-answer and clarification contrasts. | CC-BY4.0; access requires acknowledgement. Not downloaded or admitted here. Its function calls do not teach native pending/cancel/stale-result timing without additional episodes. |
| Authored concepts and domain tasks | Databases, networking, programming, basic statistics and scientific explanations with source-backed rubrics; later domain-specific modules. | Use authoritative, permitted source passages and executable checks where possible. No arbitrary YouTube scraping, private data or blanket textbook-license assumption. Teach uncertainty and retrieval for changing knowledge. |

Start the broader text tranche at approximately10,000–30,000 verified examples, not millions. This is a proposed engineering scale, to be adjusted from coverage, learning and throughput—not a research-derived requirement. Select roughly1,000–3,000 distinct situations for eventual native audio conversion after textual correctness/routing labels pass. Generate voiced subsets in coherent batches under existing permissions and actual API budgets; this plan does not increase external-service spending. Use approved British/RP/Northern/Welsh/Scottish/Irish preferences and retain their saved owner ratings. Do not assume a voice-library label guarantees the accent or naturalness of a generated clip.

Each task record needs provenance/license/revision, family and difficulty, correct concise response or expected action, rationale available to the verifier, required context/arguments, executable checks or cited rubric, and split ancestry. Timed audio episodes additionally need original WAV hashes, voice/model/settings/receipt, measured words, actual worker events and causal response timing. Split by task family before paraphrasing or speech generation. Deduplicate source seeds and paraphrases, not just exact strings.

For routing, create matched contrasts: an easy question solved locally; a similar-looking multi-step question delegated; a missing-input question clarified; an unavailable-tool question handled honestly. Include concise explanations of completed results, user corrections, unrelated conversation during work, and cancellation before/after actual completion. A keyword-triggered “always delegate technical questions” policy is inadequate.

## Training stages and objective

**Now: native post-training and retention.** Finish independent qualification of the96 scenes and build336 TRAIN plus unchanged64 DEV. The proposed matched data experiment gives both arms336 exposures/84 effective-batch4 updates from the same preserved checkpoint: old-data rehearsal control versus old240 plus new96. Match token/frame exposure where practical and report residual differences. Evaluate the whole cohorts, then continue the useful arm for meaningful complete epochs while DEV/native behaviour improves. Do not call one epoch convergence or keep repeating a failed mixture indefinitely. This is the next learning decision, not a requirement to finish the whole research programme first.

**Next: joint reasoning and routing post-training.** Introduce verified text tasks alongside native episodes. Proposed initial sampler allocation is50% native rehearsal/control episodes,30% compact reasoning/concept tasks and20% allocation/context-transfer tasks. These are sampling probabilities, not a claim of equal token or audio cost; cap and report each stream's supervised tokens, frames and repetitions. Keep every original native component represented. The mixed loader/objective must be implemented and tested before this recipe is described as executable; the current native trainer does not automatically accept text shards.

The present objective comprises native text/control supervision plus action loss and ordinary retention. Its implementation uses action0.25, KL0.1 and hidden0.01; ordinary retention is currently applied to the ordinary family. The nominal LR5e-6 has an inherited0.5 updater, so peak effective LR is2.5e-6. Microbatch1, accumulation4, global batch4, warmup16 and full native sequence backpropagation are the current reference. Log actual group rates and update norms. Changing a JSON recipe alone does not change hard-coded loss weights: wire, test and record the executable objective.

For joint training, normalise each stream's loss by its actual supervised units before weighting; mask prompt-only material and unavailable future results. Retain ordinary KL/hidden constraints and every old native objective. Acoustic training requires its own explicit speech target/loss and module choice: **feeding ElevenLabs audio to the current frozen-TTS pipeline does not train the TTS decoder or automatically bake in an accent.** Correct language and British acoustic delivery therefore have related but separate training work.

**Conditional: selective continued pretraining/midtraining.** If the untouched foundation demonstrably lacks useful concepts and supervised adaptation fails to generalise, train on a curated concept/domain corpus with general rehearsal, then repeat instruction/native alignment. Start with a measured10M–50M-token tranche and learning curves rather than a trillion-token imitation of the reports. Token counts are planning estimates, not an allocation launch. Avoid context-length expansion unless context itself is the demonstrated limitation. Keep current/changing domain facts in retrieval or the reasoning backend where appropriate.

**Later only: preference/RL refinement.** Once supervised speech and execution are reliable, preference learning can target concise helpful answers and calibrated routing; verifiable-task RL can target correct reasoning and successful delegation. Rewards must penalise incorrect confident answers, needless tools, speech during required silence and stale results. Silence alone cannot earn success. No native RL is implemented or authorised by calling failure mining “MiRL.”

## OPAL and MiRL: adaptive, accountable changes

Current646 used OPAL KEEP, API0, programversion0. KEEP means retain the existing update program; it is not an autonomous code-writing agent. A real local Git-mediated OPAL edit and fresh Adam/RNG/program restore passed on the actual ARM CPU runtime, but this is not yet native-model efficacy. [Proof](NATIVE-OPAL-LOCAL-GIT-CPU-PROOF-2026-11-07.json), [research review](NATIVE-ADAPTIVE-TRAINING-RESEARCH-2026-11-06.md).

Use the LOCAL agent as the initial decision maker; no Jev or external API agent is claimed to be active. At checkpoint/epoch boundaries, provide TRAIN/DEV aggregates, per-family recurrence, gradient/update norms, effective LR, timing and memory. Allow committed, executable changes to LR schedule, weight decay, clipping ownership or implemented loss weighting. Preserve one optimizer/scheduler owner, complete state, reversible previous program and a fresh restore. Evaluate the new program against KEEP from the same saved state when its effect is uncertain. Do not change test thresholds, held-out data, optimizer clocks or budget to make a candidate pass.

Freeze a corpus during each admitted comparison; stage new shards while training runs and switch only at a recorded boundary with a new manifest and sampler state. This is operational versioning, not a claim that datasets are mathematically incapable of changing during training. Likewise, code rewriting is possible, but must answer a measured optimization problem rather than add instability for its own sake.

Use MiRL's recurring-failure idea now: cluster TRAIN errors, identify the first causal mistake, produce verified corrected episodes plus benign contrasts, and test recurrence on separate fresh data. This is supervised failure-focused training, not a reproduction of its RL method. Limit mined examples to a declared portion of the new mixture so one recurring failure does not erase ordinary conversation. Keep weight-space surgery reserved for a localised pathological activation/loop with reversible evidence, not general knowledge acquisition.

## Capacity: what can actually be added

The698M current trainable parameters are LoRA factors across the existing language network. They increase adaptation capacity; they are not698M pretrained knowledge parameters, and do not turn the native pipeline into a frontier reasoner. LoRA matrices cannot generally be copied into another foundation with different weights, dimensions or representations. Reusable PersonaPlex/Context-Spanning work includes data, protocols, receipts, evaluations and architectural lessons; its checkpoint remains preserved.

Structural expansion is mathematically possible. [LLaMA Pro](https://arxiv.org/abs/2401.02415), with [author code](https://github.com/TencentARC/LLaMA-Pro), expanded Llama2-7B to8.3B by inserting initially identity-like blocks and training those blocks on code/math while freezing inherited ones. That supports a controlled growth experiment; it does not guarantee retention in our speech-conditioned network after the new blocks learn.

Actual Venus config read from Isambard at14:39:57UTC:36 decoder layers, hidden4096, MLP12288,32 attention heads and8 KV heads. For standard Qwen-style attention/SwiGLU matrices, one added block is approximately:

`2*h*h + 2*h*(h*KV_heads/heads) + 3*h*MLP = 192,937,984 matrix parameters`.

Thus two blocks add approximately386M and six add1.158B, before small biases/norms. These are config-derived design estimates, not an instantiated model count. Confirm exact implementation, tied parameters and trainability by enumeration before allocating a growth run.

Recommended escalation order:

1. Establish whether the current698M branch learns the broader verified data without ordinary or speech regression.
2. If evidence points to restricted adaptation rather than data/recipe, compare selective full-weight tuning or different module/rank coverage with the current adapter at matched useful exposure.
3. If capacity remains the plausible limit, create a disposable identity-initialised two-block branch. Preserve input/output dimensions, tokenizer, audio interfaces, KV-cache layer indexing and conditioning. Verify initial native equivalence, useful gradients and complete state restore, then train on enough new data to test the hypothesis.
4. Consider six blocks/>1B only if the smaller growth branch improves held-out capability without unacceptable voice latency. Additional parameters start without additional learned knowledge and increase inference work. Reject growth that merely reduces TRAIN loss.

Keep foundation replacement as an explicit competing option: if fresh matched tests show poor native reasoning or speech that persists after the targeted repair, compare an available pretrained native-duplex alternative against further Venus training on the same product tests and total adaptation cost. A stronger text backbone alone is not a drop-in replacement for the audio interfaces, streaming state and learned duplex alignment. Do not switch on a parameter count or model-card benchmark alone, and preserve the current checkpoints/data so the comparison is reversible.

Full-weight tuning may fit with sharding, but also changes the largest amount of inherited behaviour. Compare it deliberately; it is not an automatic next stage after LoRA. From-scratch pretraining would additionally recreate language, listening, speech and duplex alignment and is not the economical route to this product.

## Isambard first; GCP is a fallback, not a required distributed partner

**We are not inherently limited to one Isambard GPU.** Official specifications describe four GH200 GPUs per node and distributed PyTorch support. The award is20,000 GPU-hours, equivalent to5,000 node-hours before actual usage; it is a time allocation, not proof of a specific instantaneous quota. Inspect current account/QOS/partition limits and available hardware before scheduling. [Specifications](https://docs.isambard.ac.uk/specs/), [distributed tutorial](https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-training/).

Accounting uses the maximum allocated GPU/CPU/memory fraction. One appropriately sized GPU allocation is normally0.25 node-hours per wall hour; requesting disproportionately large CPU/memory can cost more. Even CPU-only Slurm work on Isambard-AI has a minimum GPU-sized allocation. Do not describe scheduled download staging as zero allocation cost, and do not place long/intensive workloads on login nodes. [Official accounting](https://docs.isambard.ac.uk/user-documentation/guides/accounting/).

| Work | Preferred compute and trigger |
|---|---|
| Manifest construction, verification and data staging | CPU work; bounded scheduler allocation for sustained work. Never reserve multiple GPUs for an avoidable download. |
| Current698M adapter and native evaluation | Proven single-GH200 route until the measured distributed route is admitted. Reuse cached weights/data and existing restore proofs. |
| Faster same-model training | One Isambard node,2 GPUs first, then4 if effective. One training job, serial comparison, no overlapping model jobs. |
| Full-weight tuning or larger expanded state that exceeds one GPU | Isambard FSDP/ZeRO-style sharding after a measured memory requirement; DDP alone does not reduce replicated model-state memory. |
| Isambard capacity/outage or unavailable required runtime | GCP only after current credits/eligibility/quota/price/network topology and a concrete spending/time plan are verified. Previous credit screenshots are not current purchasing authority. |

Before scaling, collect steady-state seconds/update, supervised tokens/audio frames per second, allocated GPU-hours per equal work, peak memory, checkpoint time and input wait. Current full-gradient admission measured roughly42.36GB peak for this adapter setup; new data lengths or modules can increase it. Conventional dense Adam accounting is roughly12–16 bytes per trainable parameter before activations/runtime, depending on precision/master weights. Use an actual run estimate, not only parameter count, for a larger branch.

Use two GPUs with microbatch1/accumulation2, then four with microbatch1/accumulation1, preserving global batch4. Check loss normalisation, gradient reduction, clipping, per-rank RNG, sampler cursor and OPAL state. Distributed reduction need not be bitwise equal to a single-GPU trajectory; predeclare numerical comparison tolerances, while requiring exact fresh restore within the same admitted runtime. Current one-GPU guard must be explicitly extended and tested; do not bypass it by editing a lease.

A useful initial acceptance target is at least1.5× speed on two GPUs and no more than1.35× GPU-hours for the same work, with correct restore/behaviour. These are proposed trade-offs, not measured speedups. Review one bounded throughput comparison; if it fails, return to the proven route instead of launching a kernel research programme. Selective activation checkpointing, efficient data loading and safe sequence bucketing are candidates only when measurements identify their bottleneck. Never flatten native history/causal masks to improve a benchmark.

Do not synchronise one training job across Isambard and GCP over the public Internet. Keep tightly coupled gradient communication inside one provider's suitable interconnect. Use the other provider for independent staged work or later checkpoint continuation only within scope, with provenance and preservation. GCP distributed GPU SKUs/topologies must be checked at decision time. No GCP allocation, transfer, purchase or budget increase is made by this document.

## Execution order and protection against another diagnostic spiral

1. **Now:** finish the independent96-teacher review, native cache and336-TRAIN admission. Preserve the failed13/16 reviewer; use the separately prepared32B reviewer once, without changing controls. Finish this research decision record and keep the next learning work moving.
2. **Alongside CPU preparation:** freeze the fresh reasoning evaluation specification and compile licensed, verified text candidates; do not spend days collecting a giant corpus before the next run.
3. **Next learning run:** execute the complete data comparison, then continue the useful recipe. Enable the proven local OPAL decision path only after native state equivalence, and record every applied program. Do not replay646's saved updates.
4. **Before long expanded-data training:** make the single-node2-GPU throughput comparison; scale only if it pays. Broaden native reasoning/routing coverage from the verified text pool, maintaining speech and interaction rehearsal.
5. **At the first full reasoning/native readout:** choose data/retention, acoustic repair, routing, selective midtraining or capacity using the decision table. Do not automatically enqueue all research branches.
6. **Product acceptance:** full reasoning, native speech/control, actual async worker and released/candidate WebSocket/PCM tests; blind British naturalness ratings and measured end-to-end latency. The inspected32/64 regressions remain mandatory but are not sufficient alone.

Each run record must include source/base/checkpoint hashes, dataset version/splits/exposure, actual effective recipe/loss code, trainable counts/modules, optimizer/RNG/OPAL/sampler states, complete per-family results, elapsed/allocation cost, failure reason and backup hashes. A30-minute hypothesis review or20-minute infrastructure review asks whether to fix one concrete cause or change approach; it does not interrupt valid training, transfer or evaluation. No new diagnostics are required unless their result can change the next action. The goal stays active until the complete voice product is demonstrated.
