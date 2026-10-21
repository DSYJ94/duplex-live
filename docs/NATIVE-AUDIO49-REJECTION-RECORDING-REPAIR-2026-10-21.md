# Preserve failed or uncertain transition state

Watcher7589159 verified the complete restored candidate7590052, then `sbatch`
rejected native10 with `AssocGrpCPUMinutesLimit`. The old watcher exited1 while
leaving `watcher_terminal` unset and the native10 attempt marked submitting. That
prevented the separate terminal-backup observer from progressing until independent
accounting checks proved no job had been accepted and LOCAL reconciled the receipt.

The new handler records a FAILED watcher before propagating a transition exception.
It preserves the submission attempt and every possible accepted/held job ID. It
never clears an attempt, retries a submission, releases an uncertain job or weakens
checkpoint/resource/deadline gates. The existing SlotBusy wait remains separate.
Quota remains an external limit; this change does not grant compute access.

The regression drives the actual watcher with a persistent temporary receipt and
simulates both a rejected submit and a timeout after an accepted held ID. Both
failed with missing terminal state before the change; all7 controller tests pass
locally after it. The existing tests also retain unknown-state and duplicate-submit
protection. These are model-free orchestration tests; full288-test CPU execution
on Isambard is pending resource access. The future readiness watcher count is
updated288 to include both newly added worker/controller regressions. The completed
741872f payload and all original reports remain unchanged.

Slurm defines this reason as an aggregate association CPU-minute limit over past,
present and future jobs. See the [official reason-code reference](https://slurm.schedmd.com/job_reason_codes.html).
Native model behavior is still untested for this checkpoint; full numerical restore
and preserved artifacts do not qualify a model for adoption.

The same7 model-free tests independently pass on Isambard host Python3.6.15 from
immutable d08ef7a09f17dbb3600827aa989bfe23e77f4bc7 in0.081s. No model/GPU was
loaded and no Slurm job was submitted. The exact host validation receipt is backed
up with size/SHA checks. Complete Git history bundle source-main-d08ef7a.bundle is
verified. This validates controller portability and failure handling, not native
inference or the full288-test model/state suite, which remains unrun under the quota.
