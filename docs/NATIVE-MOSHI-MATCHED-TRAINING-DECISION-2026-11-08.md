# Native training: the next major decision

The completed weight-isolation test is in [the full measured result](NATIVE-CONTEXT-INITIALIZATION-RESULT-2026-11-08.json). All64 outputs,17,506files/537,470,958bytes and four driver-link targets are backed up and verified. The old native collection is also fully verified on this Mac and the new allocation:14,252files/30,039,768,329bytes. The newly encoded48 ordinary TRAIN scenes retain approved ElevenLabs British source voices and every original word/sample.

| Same engine, voice, sampler and32 inputs | Raw PersonaPlex | Released Context Spanning |
|---|---:|---:|
| Generated-word/audio agreement plus peak gate |11/32|18/32|
| Raw peak gate |32/32|32/32|
| Necessary silence and release |0/4|0/4|
| Silence while paused |0/4|0/4|
| Basic questions that triggered RET |0/12|12/12|
| Any RET in the32 scenes |0/32|32/32|
| Supplemental basic core-answer matches |10/12|4/12 after unfulfilled handoffs|

The last row is a manual review of generated text, not speech or general reasoning qualification. Raw PersonaPlex also adds false or questionable explanations about wings and leverage; a correct core clause does not excuse these. Context Spanning was tested without its backend on purpose to isolate weights, and its12/12 basic handoffs mean the resulting wrong/incomplete answers do **not** prove that the connected stack lacks knowledge. Its pretrained behavior is strongly backend dependent. Neither condition satisfies the user's desired combination of competent native answers, calibrated delegation and required silence. Both have approximately34ms p95 steps for an80ms clock, with64/65 startup/other steps above80ms; this is not end-to-end live latency. The current comparison does not certify a British voice.

The old native model was PersonaPlex **overlaid with the full released Context Spanning checkpoint**, then our1.8944M adapter. Its adapter cannot simply be attached to raw PersonaPlex and claimed equivalent. All old weights, full states, source, original receipts and failures stay preserved.

## Recommendation: learn before choosing a foundation

Run one matched **two-by-two training experiment** using the existing Isambard award:

- Initial weights: raw PersonaPlex versus the released Context Spanning checkpoint.
- Adaptation configuration: existing last-four-layer/rank8 scope (1,894,400trainable coefficients) versus all32 temporal layers/output+FFN projections and text head/rank64 (105,070,592). This changes coverage as well as rank, so it is not a pure parameter-count experiment. Packed QKV remains unadapted.
- One scheduler job at a time, two GH200s: the two initializations train in parallel, with the two capacity configurations serial on each GPU. This avoids idle GPUs while a larger branch runs and requires no distributed optimizer implementation.
- Four complete coverage epochs initially,196new updates per model under the proposed49-step schedule; four separate fresh training lineages, not a claim that one model receives784updates or a replay/resume of old520/528. At most96aggregate GPU-hours across bounded jobs and verified post-update resumes, inside the20,000-hour award; no purchase, GCP compute or external model API calls.

The49-step coverage cycle includes every original28 actual-history component, four reference components, old25/new24 teacher components,10 word/14 clock anchors,49 joint-audio components and49 stream auxiliaries, control40, reasoning32 and newordinary48. Small groups cycle explicitly; per-member visit corrections retain the declared complete-cycle objective mass. This is a new stratified accumulation schedule, **not** numerically equivalent to the old154-component full batch every update. Original folds, words, audio, receipts, KL1/hidden10 and protected teacher terms stay intact. No old gradient report is relabelled as approval for a new base or105M parameters.

Address the measured sparse request-target mass with a separately normalized positive/negative RET decision auxiliary drawn from **unchanged existing TRAIN target positions**, alongside the full native text/acoustic losses. Its initial proposed coefficient is8; verify the exact decision mask before training. Sparse target mass is a measured risk, not an established sole cause of failure. Do not invent labels or use inspected evaluation questions as TRAIN. The source-complete ordinary48 teaches concise British speech/basic reasoning; the older reasoning32 already teaches spontaneous hard-task delegation, cancellation and replacement, albeit within eight matched optimization problems.

Proposed initial AdamW LR2e-6/warmup16/clip1; publish actual optimizer defaults, loss coefficients, sampler counts and real training precision before launch. OPAL changes need versioned executable programs, complete-state restore and recorded epoch-boundary evidence. For the matched initial comparison, apply a common declared optimizer policy and report any differing adaptive trajectories honestly. MiRL-style data proposals come from repeated TRAIN failures and enter only as separately qualified versions. No direct weight surgery or full-weight pretraining in this phase.

Only required admission work remains: native conversion/capture of all48 new scenes, valid original-data mounts, zero-expansion/native gradient checks for the actual configurations, materialized Adam and independent RNG/program/state restoration, and an honest throughput fit for a complete schedule. Stop an unworkable branch after two failed admissions instead of another zero-update retry chain. Once admitted, train complete epochs, save immutable states and avoid pausing valid training for observation or copyback.

Use full native regression results at declared boundaries, preserving original folds and acknowledging the regression cohort has already been inspected. Preserve the fresh20 for the unchanged final qualification process. Compare the public baselines and trained candidates with the **same real backend** during final delegation evaluation; the disconnected comparison above cannot be reused as a connected-stack baseline. Require genuine pending work, cancellations, result grounding, speech and ordinary reasoning before adoption. Native64/20 gates and independent restore remain required in their applicable scopes; no passing subset or gain reduction.

The alternative is backend-first: connect and test the existing Context Spanning stack before changing the learner. That can establish its intended system behavior faster, but does not itself teach the foreground to answer basic questions directly. A switch to a different foundation or full tuning above1B trainable weights is deferred until this choice is made and useful learning results exist.

Status: proposal for the owner's major-direction decision. No new training is running or claimed admitted. Parameter helpers exist and CPU tests pass; the new integrated training runner and its actual native admission are not yet complete.
