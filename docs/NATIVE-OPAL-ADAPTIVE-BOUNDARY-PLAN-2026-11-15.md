# Activate result-driven OPAL decisions at evaluated checkpoint boundaries

Status: owner-requested plan; NOT deployed to the live worker. Current training remains valid and uninterrupted. No new provider calls, GPU job, optimizer update, dataset change or source mutation is performed by this plan.

## Why KEEP is still active

This is an implementation/configuration gap, not a conclusion that adaptation is undesirable. `training/personaplex/distributed_opal.py` constructs `NativeOpal(... backend='keep', agent_budget=0.)`. There is no active decision author/controller behind it. The existing `training/venus/opal_supervisor.py` verifies and transports committed decisions; it does not diagnose results or write improvements by itself. `NativeOpal` also defaults to update16/32/64/128 decisions, whereas this run evaluates after49/98/147/196. Changing a label or writing a decision file cannot activate a backend that was never connected.

The initial fixed-recipe run provides a reference, but permanent KEEP is not the desired end state. An active adaptive controller can legitimately select KEEP from evidence; it must also be able to apply a justified, verified change. These are different from a hardcoded KEEP backend.

## The live-worker constraint — do not promise an unavailable switch

Actual worker source3c6f699e85adb26ba764e473ee491c041c62f6db is immutable. It performs DEV64, publishes a checkpoint and backup at49, then immediately starts the next update. There is no epoch-yield file, signal handler or runtime backend switch. Its supported cooperative yield is the scheduler-deadline checkpoint continuation. Source: `training/personaplex/expanded_runner.py`, `expanded_launch.py`, `expanded_guard.py`.

Consequently, guaranteed application immediately after49 AND literally uninterrupted execution are not both available in this worker. Do not hot-edit program/source files, alter the lease/deadline, suspend/scancel a healthy worker, or race update50 to pretend otherwise.

The no-interruption route is:

1. Prepare the boundary-aware continuation and decision system off the live worker.
2. At the next complete evaluation (49), produce and validate a real adaptive decision from its evidence while the existing worker continues.
3. Apply it at the first successful cooperative checkpoint handoff, using the LATEST complete checkpoint, not an older49 checkpoint if further updates have completed. If no handoff occurs before196, application moves to the next substantive continuation; say so clearly.

If the owner instead prioritises exact49 application, a controlled checkpoint restart is a distinct operational trade-off. This plan does not silently interpret “do not interrupt valid training” as permission to terminate it. Report integration readiness before49, and distinguish DECISION_READY from APPLIED. No intervention on current8478956 is scheduled here.

## Boundary transaction to implement in the next continuation

Evaluation and decision occur after completed updates49/98/147/196 (or the next remaining boundary on a later continuation). A final196 decision prepares the next phase; it does not authorize extra updates in this196-update scope.

1. Finish all64 DEV examples and aggregate TRAIN metrics over the COMPLETE preceding epoch, with coverage/denominators recorded. Publish the unchanged pre-decision checkpoint and independent hash backup, including all four rank-specific Adam/OPAL/RNG states.
2. Rank0 creates an immutable decision request bound to update counter, checkpoint/report hashes, old/new recipe identities and data/fold version. All ranks enter the same deliberate boundary barrier.
3. A LOCAL result-driven controller chooses a validated recipe proposal. Initial automated operation uses a precommitted, parameterised program catalogue and explicit policy: no paid API, no assumption this chat is awake every120seconds. Current Codex can review/write additional committed candidates between boundaries. The Git broker alone is not this controller. A separate LLM/Jev/API agent remains optional future work with its own payload/cost/restore contract.
4. Validate syntax AND execution/state preservation, bounds, scheduler/clipping ownership and candidate source deployment before application. The GitSupervisor's current AST parse is syntax checking, not a security or numerical proof.
5. Broadcast ONE exact program hash and decision to all ranks. Restore and verify all common Adam fields, each rank's OWN RNG/diagnostics and native parameter fingerprints BEFORE the intentional program transition. Never broadcast rank0's RNG over the other ranks.
6. Apply the program/recipe transition once, at unchanged update count, then save an after-transition state and receipt. OPAL program version and policy state may change intentionally; do not falsely call the entire old/new OPAL state an exact same-program restore. Adam moments, parameters, data cursor and learning progress must not reset.
7. Resume at the next unsaved update. Log actual effective LR as well as nominal scheduler LR, reason, metrics, program source/hash, all-rank agreement and next-step state. Require fresh-process restoration of the new program state before a later resumption.

Do not trigger an automatic OPAL step-based request at49 BEFORE DEV49 exists. Implement and test a boundary-owned decision adapter using the pinned upstream interface; until verified, it is an integration task, not a claimed feature. Disable/migrate the old16/32/64/128 decision schedule explicitly while preserving its historical ledger/counters.

Decision waiting is bounded (initial target120seconds). Missing, stale, invalid or late proposals leave the CURRENT valid recipe running and defer a decision; they do not stop the job or accept unverified code. The automatic local policy avoids needing continuous LLM polling. No valid response is not evidence of adaptive operation; record fallback honestly.

## First recipe changes and selection evidence

Begin with scheduled-LR multipliers, preserving AdamW, clipping limit1, warmup/cosine clocks, all current objectives, all323 retained components, all1008 TRAIN exposure and64 DEV separation. Candidate factors0.5/0.75/1.0/1.1 are initial engineering bounds, not research-established optima or a preselected change. Do not force a change merely to demonstrate activity.

| Complete evidence | Candidate response |
|---|---|
| TRAIN improves while DEV or retention worsens beyond measured variation | Reduce effective LR; retain rehearsal. Classify the failing family before broader changes. |
| Both DEV text/audio improve, retention is stable and numerics are sound | Retain or cautiously raise the multiplier; require explicit evidence and bounds. One post-baseline DEV point does not prove a plateau. |
| Lower teacher-forced loss but generated words, reasoning or control fail | Do not reward the loss alone. Queue native behavioural evaluation and targeted data repair; optimizer tuning is not established as the cause. |
| Missing/contradictory metrics or an ambiguous trade-off | An evidence-bearing KEEP/defer decision, not silent permanent KEEP. |
| Persistent base knowledge/capacity gap under matched controls | Propose selective/full tuning, domain mid-training or function-preserving growth as a separate measured branch, not a silent optimizer rewrite. |

Use separate DEV text/audio metrics, retained-group objectives, four-reference KL/hidden drift, finite gradients, clipping statistics and coverage. Current DEV64 is teacher-forced; it DOES NOT establish reasoning correctness, natural British speech, silence, calibrated delegation, connected-worker success or latency. Add separately qualified DEV behavioural measurements to later decision packets; never expose final test answers to optimization. Do not infer an appropriate clip threshold merely from the raw norm of a summed multi-objective loss.

Loss-balance, decay, clipping and data-mixture adaptations can follow once their own ownership/retention tests and versioned contracts are implemented. Preserve the current retention floors and complete coverage initially; do not remove objectives or mask failed families to improve a mean.

## Reuse the existing OPAL work correctly

`training/venus/opal_programs/lr_half_v1/updater.py` and `evals/harness/test_venus_opal_lr_half.py` already demonstrate temporary scaling of the nominal scheduled LR without compounding or resetting Adam. Reuse that pattern, not the file unchanged: its CLIP_GRAD_NORM=1 would DOUBLE-CLIP under the distributed runtime, which already owns collective clipping. A distributed candidate must not clip again.

Keep one scheduler owner. A direct LR assignment can be overwritten by the existing LambdaLR step. Either use the tested temporary-effective-rate pattern with `follow_user_scheduler=True`, or explicitly migrate to program-owned scheduling with `follow_user_scheduler=False`; never do both. Record the multiplier/policy history and effective rates in the full OPAL state. First implementation should use the former, simpler pattern.

Checkpoint changes of source/backend/recipe require an explicit transition lineage. The current runner requires exact recipe/source restoration, so a new source cannot simply be pointed at a saved checkpoint and called a same-state resume. Restore the predecessor under its original identity first, verify it, then append the intentional transition. Preserve old program files, scheduler clocks, moments, observations, decisions, cost ledger and diagnostics. Returning to an earlier recipe uses the LATEST valid weights/state; do not replay completed updates from an older checkpoint without a separate explicit rollback exception.

## Required tests before activation — focused, no infrastructure spiral

- Deterministic policy fixtures: improving, regressing, missing, noisy and conflicting metrics; no test/fold leakage or guaranteed ADAPT verdict.
- Real optimizer/scheduler regression: multiplier persists without compounding, each update advances once, clipping once, moments/counters preserved, all losses/coverage unchanged.
- Real multi-rank transition: one author, unanimous program hash, distinct RNG states preserved; stale/replayed requests rejected; timeout continues current recipe.
- Exact fresh-process restoration before and after a zero-update program transition; no dummy native optimizer steps or saved-update replay.
- Boundary ordering: complete DEV -> checkpoint+backup -> decision -> versioned transition -> next update. Existing valid worker remains unmodified.

These tests are preparation for a new continuation, not permission to re-run completed1072 captures or196 historical training.

## MiRL, RL and the wider programme

Keep `NATIVE-RL-AND-RESEARCH-INTEGRATION-2026-11-11.md` and `NATIVE-TRAINING-CRITICAL-REVIEW-2026-11-15.md` active alongside this plan.

- MiRL-style targeting: mine repeated TRAIN failures, separate perception/knowledge/routing/acoustic/control causes, create qualified recovery examples and retain broad rehearsal. Version the next dataset at a boundary; do not convert DEV/final tests into TRAIN or silently change this run's full-coverage corpus.
- Native GRPO/RL: use a coherent SFT checkpoint plus real connected rollouts, verifiable correctness/dispatch/cancellation/result receipts, waveform timing/silence and retention rewards. Coherent useful attempts are required, not perfect behaviour; speech loss is not the reward. Compare against retained SFT control and preserve sampling/reference/RNG/optimizer/OPAL/reward state.
- Capacity/full tuning/mid-training: decide from matched base-versus-adaptation and family-level results.105M is a control, not a settled optimum. Keep the +0.5–1.5B growth and selective/full-LM arms in the broader plan, with suitable data and exact streaming/state/retention tests.

No RL learner, MiRL curriculum or arbitrary weight-space surgery is activated by this document. The goal is a useful adaptive training system, not a renamed KEEP backend or uncontrolled live code edits.
