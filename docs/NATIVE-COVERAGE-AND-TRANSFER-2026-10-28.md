# Native coverage and transfer assessment — 28 October 2026

Status: research and proposed next data version only. No synthesis, credential access, provider mutation, GPU work, optimizer run, or model adoption was performed for this assessment.

The present data is narrow, especially pending-work conversations, but adding more audio alone is not a demonstrated cure. Prioritize native speech delivery and runtime correctness, then expand distinct interaction timelines. Reuse existing recordings and acquired public sources before purchasing another ElevenLabs batch. PersonaPlex/Context-Spanning work carries over as evidence, source material and engineering knowledge; its trained adapter tensors do not carry directly into Venus.

## Actual current coverage

Counts below were recomputed from **TRAIN and DEV only** in [repair source V2](../evals/fixtures/venus-control-repair-source-v2-20261027.json). A unique timeline here means exact equality of the complete `user_turns` array, independent of private-world values. This is a conservative diversity diagnostic, not a semantic clustering score.

| Family | TRAIN scenes | DEV scenes | TRAIN distinct user timelines | DEV distinct user timelines | Concrete limitation |
| --- | ---: | ---: | ---: | ---: | --- |
| Quiet until cue | 56 | 12 | 56 | 12 | Short authored instructions; no general proof of silence under natural overlaps or longer conversation |
| Delegate/result | 48 | 12 | 12 | 4 | Different hidden values repeat the same user interaction; simple marker lookup rather than varied real reasoning |
| Cancel pending work | 40 | 8 | 12 | 3 | One pending lookup and cancellation, little lifecycle/race diversity |
| Ordinary local answer | 32 | 12 | 32 | 12 | Small set of short factual/common-sense answers; no broad reasoning or contextual conversation coverage |
| Pause/resume | 24 | 4 | 24 | 4 | Scripted two-step narration, pause and continue; limited interruption/correction/task-switch behavior |
| Question while pending | 16 | 4 | **4** | **1** | Narrowest interaction coverage: four TRAIN conversations and one DEV conversation repeated across worlds |
| Total | **216** | **52** | **140** | **36** | All authored synthetic; no owner conversations |

TRAIN has 82 `split_group` values and DEV 26. These group counts are neither scene counts nor unique-timeline counts. Counterfactual worlds are valuable for testing result grounding; preserve them, but do not mistake four values behind one question for four interaction patterns. The source JSON's static status/zero-update fields describe its authoring snapshot. They do not supersede later approval, training and evaluation receipts.

The [experiment tracker](NATIVE-EXPERIMENT-TRACKER-2026-10-28.md) records five 152-scene TRAIN epochs for Venus190 and one complete 216-scene epoch for Venus244. The latter adds 54 updates, not another five epochs. Its matched full52 DEV loss improved 0.9654868→0.8425183, while quiet loss worsened 0.9438%. This is evidence of fitting some teacher targets, not reliable autonomous behavior. Source: [reactive54 results](NATIVE-VENUS-REACTIVE54-RESULTS-2026-10-28.json).

The independent [complete paired64 review](NATIVE-VENUS-PAIRED64-REVIEW-2026-10-28.json) contains all 32 cases under each condition: generated-word/audio agreement passed **1/32 for parent190 and 7/32 for paced244**; all 64 passed raw peak checks. This agreement metric compares emitted words with heard speech, not task correctness or naturalness. Quiet diagnostics still include voiced chunks before the release cue; a quiet chunk is not independently judged intelligibility. Neither candidate is qualified or adopted. Do not select a passing subset or infer that silence/delegation/ordinary task accuracy improved by the speech-agreement delta.

## What is available versus actually trained

[Acquisition receipts](PUBLIC-NATIVE-DATA-ACQUISITION-2026-10-25.json) and the [project reset](PROJECT-RESET-2026-10-25.md) record 5,092 files / 14,771,623,383 bytes:

| Existing source | Measured inventory | Potential use | Boundary |
| --- | --- | --- | --- |
| DailyTalkContiguous | 2,541 dialogues, 21.674 hours | Natural phrasing, conversational rhythm, voice-retention candidates | Acquired, not admitted into Venus training; recorded CC-BY-SA-4.0 training/release use remains unapproved |
| Glaive function calling V2 | 112,960 text rows; 63,218 with function-call marker | Tool argument generation, result fidelity, ordinary-versus-tool decisions | Synchronous text; requires verified asynchronous spoken timelines and semantic filtering |
| UltraChat200k | Three `train_sft` shards, 732 MB; physical rows not measured | Ordinary conversation and retention candidates | Not the entire named dataset, not trained, not native acoustic targets |
| ElevenLabs source voices | 174 recordings, nine voice choices; complete word audit 169 pass / 5 fail | Existing reference-conditioning trials; possible future approved acoustic targets | Entire cohort remains unapproved for training; failed originals preserved |

The current control adapter freezes the audio encoder, speech generator and base language weights and has **no acoustic training loss**. Extra ElevenLabs user audio can improve input coverage but does not directly teach assistant timbre or prosody. Extra assistant recordings cannot change frozen acoustic weights through a nonexistent acoustic objective. A native voice-reference change is a distinct conditioning experiment and can help without acoustic training; it still needs a paired native test.

Owner feedback is separate from recognizer correctness: nine American clips average naturalness 2.44/5, clarity 5/5 and delivery 3.33/5. Warm conversational samples average naturalness 2/5. George is the preferred direction but has zero numeric owner ratings. A preferred reference is not an adopted production voice. Source: [owner feedback](NATIVE-OWNER-VOICE-FEEDBACK-2026-10-27.json). The [voice plan](NATIVE-LIFELIKE-VOICE-PLAN-2026-10-27.md) preserves earlier failures and the closed-provider provenance amendment; its historical lower audit snapshots must not override the complete174 audit.

## Proposed data version: coverage V3, category specifications only

This is a reviewable **proposal**, not a collection request, approved mixture or empirically optimal size. Keep V2 immutable. Before authoring new texts, freeze a separate group-disjoint TRAIN/DEV manifest and require an independent author/verifier to protect evaluation material. Count distinct user/event timelines first; list private-world expansions separately.

| Priority | New category | Proposed distinct TRAIN/DEV timelines | Required variation and verification |
| --- | --- | ---: | --- |
| 1 | Useful conversation while work is pending | 48 / 12 | Multiple follow-ups, clarification, progress requests, changing ordinary topic, result arriving during/after speech; verify responsiveness plus result fidelity and nonblocking execution |
| 1 | Returned-result delivery | 32 / 8 | Multiword values, names/numbers, different result lengths, absent evidence and conflicting results; hold user audio fixed across 2–4 private worlds; each factual token follows actual receipt |
| 1 | Silence and resumption | 24 / 6 | Explicit silence versus allowed acknowledgement, later cue, nonrelease mention, varied gap/turn lengths; native speech onset/offset and no filler during required silence |
| 2 | Interruption/correction/task switching | 24 / 6 | Genuine overlap, stop/revise, correction mid-sentence, continuation without restart; measure yield and correct resumed content |
| 2 | Ordinary and contextual conversation | 24 / 6 | Short reasoning, self-correction, uncertainty, clarification and multi-turn reference; correct local answers without unnecessary delegation |
| 2 | Async failures and cancellation races | 16 / 4 | Slow/failed worker, timeout, cancel before/during/after result, stale/duplicate result; no cancelled or cross-task facts spoken |
| Proposed first wave total | Six categories | **168 / 42 distinct timelines** | World expansion, acoustic variants and any later size increase require separate manifests and accounting |

These are additional category targets, not silently added optimizer scenes. For counterfactual groups, freeze the same user turns and audio across values; change only private result content and the causal response. Keep related paraphrases, private worlds, worker histories and audio variants in one split group. DEV can guide development; no exposed evaluation item becomes new TRAIN. This audit did not inspect actual HOLDOUT scene rows, but a builder-source read displayed HOLDOUT construction templates. Therefore no detailed new episode texts/answers or new qualification set are authored here; future sealed evaluation requires an independent author with an explicit exposure ledger. The complete64 observed set is already diagnostic material, not fresh qualification.

Use existing source audio when exact text and timing match; rerender only changed phrases or diagnosed failures. Do not spend on silence. For new assistant speech, first compare already recorded George/reference options under a fixed native protocol. Request additional ElevenLabs only when a named missing phonetic/prosodic/interaction target cannot be covered by accepted existing sources, or when a failed exact-word source needs a versioned replacement. Keep the five failed records excluded and preserve every original. Require exact source/provider/model/settings hashes, independent words/peaks, timing checks and human listening before acoustic admission. No blanket additional multi-accent collection is justified by current evidence.

## Determinants that data volume does not isolate

1. **Runtime transport and playback.** Correct chunk ordering, buffering, native flush/end-turn handling, encoder reset between scenes and actual microphone/speaker operation must be verified on the current path. Earlier source review diagnosed whole-sentence/TURN_EOS pacing and an encoder-state leak; corrected teachers and caches do not prove live transport quality. Freeze identical runtime/precision/decoding when comparing base and adapter.
2. **Trainable scope and objective.** Current rank64/alpha64 adapter changes only final-layer q/v: 851,968 values. Text/control CE plus ordinary KL0.1/hidden0.01 retention cannot by itself demonstrate optimized acoustic naturalness. Widening layers/projections or adding acoustic loss changes the experiment and cache validity; it needs a measured numerical/behavior comparison, not an assumption that more parameters fix it.
3. **Speech conditioning and pacing.** Reference voice, language hidden-state distribution, word-unit timing, speech unit termination and streaming synthesis affect intelligibility even with frozen speech weights. Native speech, not generated text alone, must be scored.
4. **Exposure and optimization.** One corrected216 epoch may be insufficient, but no controlled ablation establishes that. LR, warmup/cosine schedule, loss balancing, ordinary retention and adapter coverage are alternative explanations. Compare complete epochs and fixed DEV families; do not confuse different scene-mixture batch losses with convergence.
5. **Real async integration.** Synthetic lookup receipts exercise mechanics. They do not establish the reliability/latency/intelligence of the intended reasoning backend or cancellation propagation through it. Trace task identity, result visibility and cancellation all the way into spoken output.
6. **Evaluation quality.** Word/audio agreement, correct answers, silence, overlap response, real backend success and blinded human naturalness are separate gates. Aggregate loss, unclipped audio and restore proof cannot replace any of them. Measure per-family outcomes, latency distributions and complete paired recordings before opening a new sealed qualification set.

Sequence: finish the current live-path diagnosis; freeze a trustworthy base-versus244 native comparison; complete source/native voice review; prepare and verify the proposed TRAIN/DEV expansion; then choose one bounded training hypothesis. Do not repeat optimizer work while transport or native speech delivery remains unexplained. This report makes no unique-cause claim.

## PersonaPlex / Context-Spanning transfer into Venus

**Direct adapter transfer: no.** LoRA adds `B @ A` to particular base-model projections; its meaning depends on the exact base weights, module identity, hidden basis and vocabulary, not just matching matrix dimensions.

The inspected Context-Spanning Moshi loader at source `79df302fa2820cae10c61bb2b7f302c52edb6ed2` specifies hidden dimension4096, 32 temporal layers, 32 attention heads, text vocabulary32000, sixteen audio streams and a separate six-layer depth transformer. The previous [PersonaPlex adapter installer](../evals/personaplex_adapters.py) targets final-eight-layer attention output/feed-forward linears plus the text head, using rank8 `lora_A.weight`/`lora_B.weight`; native QKV is a raw parameter. Context-Spanning's own serialization and span sequence conventions differ again. Source: [pinned upstream loader](https://github.com/mindlogic-ai/Context-Spanning/blob/79df302fa2820cae10c61bb2b7f302c52edb6ed2/contextspan/moshi/models/loaders.py), [prior seam review](research/2026-10-15-context-training-seams.md).

Venus is a **9B multimodal system with a Qwen3-8B language backbone**, not a 9B Qwen language core. Its acquired Audio config specifies hidden dimension4096, 36 language layers, 32 query heads, 8 KV heads and head dimension128, with discrete S3 speech tokens and a separate speech path. From these config values, q_proj is4096×4096 and v_proj1024×4096. The current final layer35 rank64 factors are A64×4096/B4096×64 for q and A64×4096/B1024×64 for v; total851,968 agrees with the trainer's explicit parameter assertion. Sources: acquired `runs/public-data-20261025/model-metadata/venus/Realtime-Venus-Audio/config.json`, [current adapter code](../training/venus/lora.py), [trainer](../training/venus/train.py), [official model details](https://huggingface.co/inclusionAI/Realtime-Venus#3--model-details). Shape derivation is an inference from config and installer, not a fresh model load.

Even where width4096 matches, the learned coordinate basis, projections, layer meanings, tokenizer and audio-code representation do not. Renaming keys, reshaping matrices or copying Adam moments would be invalid transfer. Do not import old optimizer/RNG state, Moshi hidden-state/KV/codec caches, span token IDs, persona prefixes or text/audio losses into Venus. Exact continuation is meaningful only within its own architecture and identity.

| Prior asset | Reuse in Venus | Required adaptation |
| --- | --- | --- |
| Original approved TRAIN audio/text, receipts and split manifests | Yes, subject to exact source permissions and relevance | Re-encode with Venus tokenizer/audio pipeline; rebuild native timelines and caches; preserve fold/approval provenance |
| Native event protocol and result-grounding principles | Yes | Map real arrival/cancel events to Venus delegate/feedback API; preserve causal visibility |
| Harness, independent transcripts, failure archives, restore/backup discipline | Yes | Revalidate timing, speaker layout, native action semantics and acceptance measures |
| Paired private worlds and leakage tests | Yes | Same user input across worlds, group-disjoint folds; new architecture-specific native tests |
| Accepted teacher behavior as distillation material | Potentially | Convert approved TRAIN-only outputs into semantically checked Venus targets; old rejected model is not an unquestioned teacher |
| LoRA weights, optimizer moments, hidden states, Mimi codes/spans | No direct transfer | Train new Venus adapters; regenerate model-specific representation; an explicitly trained cross-model mapping would be a new unproven project |

The work was not wasted: the strongest reusable product is the audited data/protocol/evaluation system and the record of what failed. It does not entitle either model to a success claim.
