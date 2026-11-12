# Interpreting the matched native evaluation

Job 8352856 evaluates six conditions: untouched PersonaPlex and Context Spanning, plus each foundation with its completed 1.89M and 105M adapters. Each condition generates the same 32 existing cases, preserving original PCM, input hashes, released persona/voice/sampling and raw failures. This cohort has been inspected previously and is a regression comparison, not fresh generalisation evidence.

## What the current job establishes

- Complete native generation for all 192 outputs, with no optimizer updates.
- Independent ASR agreement with the model's generated text, at unchanged WER <=0.05; raw waveform peak <=0.995. Agreement does not establish answer correctness.
- Necessary cue-silence and pause measurements from native audio frames. Passing these counters does not alone establish complete pause/resume behaviour.
- Recorded RET events and native text, which permit subsequent routing review. There is no connected reasoning worker in this job.

## Review after retrieval

Review all six complete conditions side by side, retaining case-level evidence and denominators. Score native factual/basic reasoning independently from ASR fidelity. Identify baseline-correct/candidate-wrong regressions separately from baseline failures. Inspect explanations as well as short answer clauses. Do not count a grammatically clear false explanation as a success.

Review routing for appropriate direct answers, clarification and handoff, with completeness of the delegated context. A RET event or spoken promise is not a successful executed task. Context Spanning's disconnected handoffs must not be interpreted as demonstrated connected-backend incompetence. Conversely, delegation on every easy question does not satisfy the requested native competence.

Listen to complete output recordings for intelligibility, naturalness and British-English fit, preserving failures. The released voice is not certified British, and teacher-forced loss cannot certify the accent. Case timing is recorded, but this is not a real microphone/WebSocket end-to-end latency trial.

## Decision rule and next work

No adoption from the aggregate speech score or training loss. Use the full comparison to distinguish foundation limitations, training regressions, adapter-capacity effects and speech/runtime issues. If both trained variants regress relative to their own base, inspect data/retention/objective balance before simply adding parameters. If native answers are adequate but hard-task routing fails, prioritise complete-context routing and real worker execution. If answer text is sound but speech is poor, isolate speech conditioning/runtime and acoustic training rather than conflating it with reasoning.

The next product qualification still requires the same real asynchronous backend for baseline/candidate, actual task receipts, cancellation/stale-result exclusion, fresh spoken reasoning, the preserved fresh20/native gates and real live conversation/latency. Broader data, selective/full tuning, RL and parameter growth remain the conditional tracks in the existing product and Isambard scale plans. This note neither authorizes a new compute interval nor changes any acceptance threshold or training data.

Sources: training/personaplex/matched_evaluation.py; training/personaplex/matched_finish.py; docs/NATIVE-MOSHI-MATCHED-TRAINING-DECISION-2026-11-08.md. Actual submission controller: 63f121fc876394c8d63a38e22535b6a2496015f8. Training producer: 10e7e0664e47c1222e0492a8367b3c000ed5789d.
