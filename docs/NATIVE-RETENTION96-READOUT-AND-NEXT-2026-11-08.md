# Completed paired training: outcome and next decisions

Training8179044 completed successfully on8November at11:50:58UTC. Dependent evaluation8180579 completed at12:21:32UTC. Both arms independently restored parent646 and completed84 new updates, ending at730;168 total project updates do not mean a model at814. Both final independent processes restored Adam, RNG, OPAL and native logits exactly. The original source, checkpoints, receipts, losses, folds and failed outputs remain preserved.

**Decision: retain both new checkpoints for research; adopt neither.** The experiment successfully answers whether these96 additions, under the matched recipe, improve the product. They did not deliver the required native improvement. This is a useful negative result, not a reason to repeat the same run.

| Measure | Parent646 | Expanded336 uniqueTRAIN | Rehearsal336 exposures /240 uniqueTRAIN |
|---|---:|---:|---:|
| Text reasoning, all128 |92|92|93|
| Everyday arithmetic,32 |11|11|11|
| Technical concepts,24 |23|23|23|
| Context reasoning,24 |10|10|11|
| Delegation action choice,24 text questions |24|24|24|
| Speech word agreement,32 |15|12|12|
| Raw peak threshold,32 |30|27|28|
| Both word agreement and peak,32 |15|11|11|
| Correct initial lookup key,4 delegation cases |2|2|2|
| Correct initial lookup key,4 pending cases |2|2|2|
| Correct requested work cancelled,4 cancellation cases |2|2|2|
| Any actual pending work cancelled,4 cancellation cases |4|4|4|
| Necessary silence before cue and voice after cue,4 |4|4|4|

Speech word agreement compares emitted words with the model's intended user-facing text. It does not prove the answer is correct or that the whole task succeeded. All original word/peak thresholds and failures are retained. Text128 is an authored multiple-choice probe with related templates, not broad free-form or spoken reasoning. Delegation action choices do not execute a reasoning backend. The native cohort uses a real lookup worker, not a general reasoning service.

Expanded data changed **zero** of the128 selected answers versus parent. Rehearsal changed one context answer from wrong to correct; that is insufficient evidence of a general reasoning gain. All actual prompt hashes match across conditions and all384 outputs had valid letter format. Everyday arithmetic and context are concrete weaknesses; this comparison does not establish whether they came from the foundation or earlier adaptation.

## Loss improved while speech worsened

Expanded-data full64DEV CE changes versus parent: ordinary−1.08%, delegation−8.03%, cancellation−16.14%, pause−11.08%, pending−16.98%, quiet−0.08%. Rehearsal improves the five control families but ordinary worsens4.66%. This supports retaining the new data for further work, while rejecting the claim that lower teacher-forced loss established a better live model.

Both new arms gained two and lost six joint speech cases relative to parent. Complete per-case changes, ordinary generated answers/transcripts, source-report hashes and restoration receipts are recorded in `NATIVE-RETENTION96-OUTCOME-2026-11-08.json`. The earlier keyword factual screen misses the parent's reasonable lever explanation; that screen is preserved unchanged and is not a substitute for inspecting answers.

## What the evidence localises

1. **Request fidelity remains wrong.** Bookcase inputs produce `lookup bokeh`, `lookup Borgian` or `lookup markerinborg`; cupboard requests are correct. Worker receipts faithfully expose the error. Four cancellations of some work do not mean four correct requested tasks were started and cancelled. Do not equate the different names or soften the evaluator.
2. **Basic reasoning did not improve.** Examples include12+18 choosing28, and3metres minus40centimetres choosing258centimetres. The exact same errors predate this phase. Technical concepts are much stronger on this probe. No evidence yet supports adding more capacity as the first remedy.
3. **Speech is a separate blocker.** Both new arms have20 word failures; all have closed linguistic turns and at least one recorded acoustic EOS. EOS therefore does not establish correct speech. Generated funnel/telescope explanations can be reasonable while emitted speech loses endings or words. No gain reduction or threshold relaxation is applied.
4. **Acoustic parameters were not trained.** This698M LoRA updates the language backbone; TTS and audio encoder remain frozen and acoustic loss is false. This run cannot be described as acoustic British-accent training or a demonstrated naturalness improvement. The Venus paper also freezes its acoustic decoder, so this does not prove frozen TTS is the wrong architecture: language-side conditioning can still improve or damage speech.
5. **Live performance remains unqualified.** Native speaking-frame p95 is approximately2.52seconds expanded and2.58seconds rehearsal. These offline frame timings exclude the browser/network/device and are not user-perceived latency. They do not establish a usable real-time experience.

## Next execution sequence

Copyback is complete: both complete730 states (17024152706 bytes total) are full SHA/size verified on the Mac; all1816 copied training/evaluation files (271119681 bytes, zero links) pass a complete inventory/hash check. This includes all1646 native/reasoning evaluation files. All84 update/save pairs per arm were inspected, with336 unique expanded-data TRAIN identities versus240 unique rehearsal identities across336 exposures. Retain parent646 and every previous state; do not deploy either730 checkpoint to the owner trial as an improvement. Keep the Isambard queue free until there is useful work, rather than allocating idle GPUs.

Then make one decision-changing foundation control: evaluate untouched released Venus on the already frozen text128 inputs, reusing all completed646/730 records rather than regenerating them. Add a small disjoint free-answer control only if needed to distinguish multiple-choice/prompt effects from actual basic reasoning. Record the result before deciding between retention repair, targeted reasoning training, stronger delegation or a different native foundation. No pretraining-from-scratch or added billion parameters is justified by the current result alone.

Prepare the next dataset from disjoint verified arithmetic/context/technical problems and genuine routing decisions. Use the already preserved publicTRAIN source pool plus independently checked authored problems; keep official test data and the inspected128/32 out of TRAIN. Teach short direct answers, clarification/uncertainty, complete semantic task descriptions and grounded worker replies. Mine TRAIN failures and add benign contrasts using the useful MiRL principle; do not claim supervised failure mining is MiRL RL. Dataset size should follow coverage and learning, not a new arbitrary96-scene ceiling.

Address speech fidelity/voice conditioning as its own intervention using the preserved original waveforms and traces. Reuse the6November reference-cache diagnosis: the reference is retained in the prepared token-to-wave cache; do not reopen the omitted-per-frame-reference hypothesis. The pinned actual source SHA b7b4070c confirms that TTS consumes text embeddings **plus projected LLM hidden states**. Current KL/hidden retention is applied only to ordinary scenes. Hidden-space shift into the frozen speech projector is therefore a concrete hypothesis worth testing for speech-space alignment, not a proven unique cause; untouched Venus also fails speech cases. A speech projector/conditioning adaptation or speech-token alignment objective is more directly relevant than adding arbitrary language layers. Preserve existing losses and treat any new objective as a separately recorded intervention. The next speech change must target the actual acoustic/text-conditioning boundary or train an appropriate acoustic component with aligned, authorised English/British data. Another unchanged language-only epoch is not the speech plan. Preserve exact reference-audio hashes, native units, original peaks and complete-cohort evaluation. Human listening remains required for naturalness and accent; no listening score is invented from ASR.

Use OPAL for recorded recipe changes at phase boundaries with exact state restoration; do not turn on unconstrained code changes inside the completed matched experiment retrospectively. Once the next data/objective is concrete, use a measured2/4-GPU single-job comparison on Isambard if it reduces time to useful learning, preserving global batch and native objectives. The award permits productive scale; no extra GCP purchase or external API calls follow from this plan.

The overall product goal remains unfinished: useful native reasoning, reliable asynchronous general reasoning delegation, faithful natural British speech and a measured end-to-end live trial are all still required. This completed phase establishes evidence for the next intervention, not GPT-live-level acceptance.
