# Keep capacity, training method and data quality under review

Owner direction, 15 November: continue autonomously toward useful training on Isambard, reuse completed outputs, preserve checkpoints and report stops promptly. Use the award productively. These questions remain active throughout the build; do not treat the present 105M adapter or dataset as a settled optimum.

## Current controlled learning run

Execution scope remains NATIVE-EXPANDED1072-EXECUTION-2026-11-14.json: PersonaPlex-family broad rank64, 105,070,592 trainable adapter coefficients, 196 NEW updates, all1008 TRAIN/64 DEV plus323 retained components; OPAL state preserved, KEEP/API0 for this coherent comparison. Old196 training and1072 completed codec outputs are not repeated. Successful submission, preparation or low loss is not evidence of learned capability.

## Questions to revisit at each complete epoch and final evaluation

| Factor | Evidence to inspect | Decision it informs |
|---|---|---|
| Adaptation capacity | TRAIN/DEV errors by family, gradient coverage, retention, unchanged failure patterns after useful exposure | Larger rank/coverage versus selective/full native-LM fine-tuning; do not infer insufficient base capacity solely from adapter coefficient count |
| Base capacity/knowledge | Untouched versus candidate, matched text/audio tasks, perception versus reasoning, direct answers versus calibrated delegation | Verified domain mid-training with rehearsal, stronger foundation, or function-preserving added blocks; growth needs new knowledge training and streaming/latency validation |
| Learning method | Correctness, full task receipts, timing, retention and live latency under matched data exposure | SFT seed first; selective/full tuning remains a real comparison arm. RL follows faithful connected rollouts and verifiable rewards, with SFT rehearsal and KL/retention safeguards. Scratch pretraining needs a justified corpus/token/compute plan, not just spare GPUs |
| Optimisation | Loss components, finite gradients, DEV regressions, restore fidelity, measured throughput | OPAL changes must be executable, versioned and checkpointed at boundaries; KEEP is a control, not autonomous rewriting. MiRL-style mining uses repeated TRAIN failures, never inspected test answers |

## Distinguish dataset purposes and corresponding tests

- **Reasoning/content:** the1072 expanded scenes pair questions with authored/programmatically checked or independently reviewed answers, including the1008-text curriculum. This teaches audio-conditioned content; it is not broad knowledge pretraining or evidence of general intelligence. Use fresh arithmetic, concepts, technical/context tasks and uncertainty tests; separate audio perception errors from reasoning with text controls.
- **Voice fidelity/quality/accent:** source recordings use approved British/RP voices and word/timing/peak/semantic checks. Native acoustic targets support learning but do not certify generated naturalness or accent. Independently transcribe generated speech and listen blind for clarity, British fit and naturalness. Frozen components cannot be claimed to have been trained; inspect the actual trainable module/loss path.
- **Turn-taking/duplex:** measured alignments and original all-frame native streams preserve clocks. Offline teacher forcing alone does not establish interruptions, listening during speech, response onset or end-to-end latency. Test real streaming overlap and interruption separately.
- **Control/delegation:** retained objectives preserve silence/pause/cancel/history and routing work. A spoken phrase or RET token is not an executed task. Test real asynchronous dispatch with full context, cancellation receipts, pending conversation, grounded results and stale-result exclusion. Broaden data with real pending-state contrasts where failures warrant it.
- **Retention:** preserve original histories, native speech objectives, KL/hidden constraints, fit15/folds and all checkpoints. Record correct base answers that adaptation damages; do not mislabel these as missing foundational knowledge.

Optimise each family with explicit losses and held-out behavioural measures. Do not average away failed silence, wrong answers, gibberish or false cancellation behind improved speech loss. Broaden licensed/authored data when coverage is missing; change recipe when good targets are not learned or retention regresses; enlarge trainable/base capacity when matched evidence supports it. Make one interpretable intervention per comparison, use adequate data and compute, and avoid repeated infrastructure-only pilots.

## Immediate execution

Storage repair1405bdd routes compiler/Triton/tmp files to unique allocated project directories. Actual contained runtime regression and project-storage C++ compilation/write test passed. Replacement8475202 submitted on fourGPUs; phase failed GPU seconds1084 carried forward. Verified1072 codec cache retains its original7ab60a5 source. Prior capture failed before any case report; partial artifacts quarantined without deleting originals. Observe this same job through native capture, gradient/Adam admission and fresh exact Adam/RNG/OPAL restore into actual updates. Do not restart valid training on an observation timeout.

## Observed implementation confound

15November: before learning, inserting adapters changed native BF16 outputs because the replacement feed-forward gate lost the published lazy compiler/fusion path. Raw repeats were exact and105,070,592 ownership was correct. Actual tiny native CPU regression reproduced max0.00390625, then passed exact equality plus nonzeroB gradients after using the same vendor compiler. Full hardware admission remains required. This is an implementation confound, not evidence that105M or the base has insufficient reasoning capacity. Do not attribute pre-update failures to training data, claim historical model results fixed, or weaken retention gates.

## First actual native learning measurement

Job8478956/source3c6f699 completed newupdate1/196 after allfour native gradient/Adam admissions and exact fresh per-rank Adam/RNG/OPAL/native restoration. Firstupdate220.0255seconds includescheckpoint publication; one update is not a representative throughput forecast or evidence of better capability. Full worst retained eleven-objective backward~393.1seconds, one worst expanded example~1.267seconds, peak allocatedCUDA~25.86GB. Use these measurements when sizing next capacity/full-tune arms and profiling retained-stream communication/checkpoint overhead; do not interrupt or silently alter the valid controlled run. Initialfull-state backup is7.99GB includingfour rank-specific OPAL/RNG states; pursue exact state deduplication later if justified, never discard unique optimizer/RNG state for smaller files.
