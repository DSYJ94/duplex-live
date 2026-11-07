# Active native voice product goal

Updated 7 November 2026 from the owner's direction: “Keep pursuing goal to improve it. Note down to not be conservative with Isambard.” This updates the working plan; the active goal remains unfinished. The historical 244-update reference is provenance, not the current training checkpoint.

Build a useful native full-duplex voice product approaching GPT-live/Thinking Machines interaction quality: natural English/British speech, useful ordinary reasoning, simultaneous listening/speaking, appropriate acknowledgements, reliable required silence, interruption and pause/resume, and asynchronous delegation of harder reasoning with conversation while work is pending, cancellation, result grounding and stale-result handling. A correct transcript, low teacher-forced loss or a working transport alone cannot establish completion.

Current foundation is the preserved native Venus pipeline and its experimental 698M adapter; PersonaPlex/Context-Spanning research, histories, data and checkpoints remain preserved. The current retained training lineage is parent526 plus saved8 updates (global534). Fourteen updates executed before a scheduler-observation timeout stopped job8149808; six later completed updates were unsaved. Recovery8156150 restores the exact saved8 state, preserves its recipe/Adam/RNG/OPAL and completes the remaining112 updates of the same120-update phase. It does not relabel the checkpoint as update14 or replay saved1–8. All old failed outputs and receipts remain available.

## Use Isambard decisively

Use the active20,000-GPU-hour award for meaningful learning and complete evaluations. Do not default to tiny pilots, repeated zero-update checks, fixed restart-count approval loops or indefinite one-GPU execution when more useful parallel compute is available. Choose resources from actual memory, throughput and experiment needs. Routine choices within the existing award do not need another confirmation merely because a previous self-imposed plan was smaller.

Respect actual cluster limits and preserve the no-overlapping-GPU-jobs rule. A single job can use multiple GPUs when justified; that is a proposed next throughput experiment, not a claim that distributed native training already works. Verify one-node2-GPU data parallelism against the proven1-GPU route, then4 GPUs if useful, keeping global effective batch4 and all native objectives unchanged. Use compute to reduce wall time and improve coverage; record GPU-hours and do not hoard idle allocations. Keep GCP monetary limits, external API/payload permissions and provider purchases separate from Isambard award usage.

## Adapt from evidence, without a workflow spiral

At existing epoch/checkpoint reviews, classify failures into source-data quality, coverage/mixture, teacher targets and stream alignment, ordinary-knowledge retention, request construction/result use, acoustic rendering/voice conditioning, runtime latency, backend orchestration, and optimizer/capacity limitations. Use actual raw audio, emitted requests, genuine worker states and complete per-family results. Cancellation is not exercised when malformed requests already returned an immediate error.

Review TRAIN/DEV trajectories, native task correctness, spurious delegation, silence/interruptions/pending/cancellation eligibility, word/audio agreement, raw peaks, latency, gradient/update norms, effective LR and throughput together. Never choose solely from aggregate CE. Record the recipe, sampler cursor, source/data/checkpoint hashes, exact objectives, actual updates, Adam/RNG/OPAL state, failures and measured results for every run.

- If TRAIN and DEV/task behavior improve with stable ordinary retention, use a substantial continuation from the latest real checkpoint.
- If TRAIN improves but ordinary/DEV behavior worsens, test better coverage, rehearsal and regularization. Do not assume that more parameters or more epochs are the cure.
- If both stall with valid data and finite useful gradients, test schedule/LR/objective balance or trainable capacity against the current recipe.
- If generated words are correct but speech is wrong, repair acoustic conditioning/rendering. Do not change correct labels to fit bad audio, conceal peaks with gain, or relax thresholds.
- If requests are malformed, teach complete valid requests and contrast ordinary tasks that need no delegation; also test the existing context-spanning reasoning path so the voice model need not reconstruct every detail unaided.

Use one focused intervention when its cause is uncertain; a small paired comparison is useful when it changes the decision. When several repairs are already diagnosed, make the necessary repairs, record them together and assess the resulting product without pretending to isolate a unique cause. The objective is a good product, not an endless sequence of one-factor microstudies.

## Use the useful parts of OPAL, MiRL and weight-space research

See [the primary-source research review](NATIVE-ADAPTIVE-TRAINING-RESEARCH-2026-11-06.md).

OPAL currently owns native wrapped updates, scheduling and resumable state, but runs KEEP/API0 with the inherited half-rate updater. Do not describe it as an active API code-rewriting agent. Use its observations to drive explicit training decisions; prepare a separate native mailbox/controller comparison when optimizer evidence justifies live code or hyperparameter adaptation. Preserve clipping ownership, complete state, finite waits and the previous executable program. API0 remains0 unless the corresponding external calls are authorized. Full code rewriting is available as an experimental direction, not an obligatory replacement for fixing data.

MiRL is Misbehavior-driven RL. Apply its useful failure-mining principle now: cluster recurring TRAIN mistakes, locate the first causal failure, create verified corrections and benign contrasts, and measure recurrence on a separate fresh cohort. This supervised adaptation is not MiRL RL. Native voice RL would need trustworthy task/behavior rewards plus resumable audio/KV/parser/worker state; do not install a terminal-agent RL stack merely to claim adoption of the paper.

Weight-space programming is a targeted research option for a localized, reproducible weight/activation pathology. Use disposable reversible variants, exact diffs, a simple baseline and full retention/behavior checks. Do not expect sparse surgery to supply missing knowledge or transfer published Qwen directions into Venus blindly.

Next data priority is substantially broader ordinary/technical conversation, non-delegation contrasts, complete requests and result-grounded replies, and cancellation with genuinely pending work. The research96-scene tranche is an experiment proposal, not a new hard size limit. Size later tranches to coverage and useful learning, preserve originals/folds/receipts and generate source audio only under applicable API permissions or verified public-data terms. Keep the inspected64 as regression evidence and freeze a distinct fresh evaluation cohort before candidate selection.

## Keep learning moving

Do not interrupt valid training, data production or checkpoint copyback to satisfy an arbitrary diagnostic timer. Review infrastructure stalls after roughly20minutes and a hypothesis after roughly30minutes: reuse existing proofs, seek a minimal real-container reproduction, make a concrete fix or change approach. Do not reopen closed kernel/precision sweeps or repeat saved updates. Bound speculative research to the unresolved question that could change the next action. Checkpoints are saved atomically after each completed update in the recovery implementation; immutable complete-epoch states and independent restores remain required.

Completion requires actual native speech/behavior qualification, natural British voice listening approval, successful complete reasoning delegation/cancellation with real pending work, and an actual released/candidate WebSocket/PCM conversation-route test with measured latency. Preserve owner privacy; do not record owner conversations or transfer credentials. Commit/push exact handoffs, keep LOCAL as the sole compute executor, and leave the goal active until the product-level requirements are demonstrated.
