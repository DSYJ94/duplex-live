# Faster short probes on Isambard-AI Phase 2

Plain words: official guidance confirms a faster reservation for short probes.
Checked 14 October 2026 against official documentation; live execution findings
are recorded below.

**Yes: the `interactive` reservation offers a separate route for short debugging and exploratory work. It is appropriate for our bounded backward and voice probes; production training stays on `workq`.** Faster starts are the intended benefit, not a guaranteed start time. [BriCS advanced guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/)

The reservation permits up to 4 nodes / 16 GPUs, has an 8-hour maximum and 30-minute default, allows 1 running and 1 queued job per user, and costs 1.5 times normal node-hour credits. Large volumes of batch work are prohibited. Request **only** `--reservation=interactive`: BriCS chooses the hidden `interactive` partition and `interactive_qos`. Explicitly overriding the partition is prohibited. These are Phase 2 rules; documentation does not promise every account eligibility or current availability, so the submission receipt must be checked. [BriCS scheduling reference](https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/)

## Commands

An interactive shell, using the documented reservation route with our account and explicit resources:

```bash
srun --account=brics.u6si --nodes=1 --gpus=1 --time=00:20:00 \
  --reservation=interactive --pty bash -i
```

An allocation for several sequential short commands uses the same reservation option:

```bash
salloc --account=brics.u6si --nodes=1 --gpus=1 --time=00:20:00 \
  --reservation=interactive
srun hostname
# Run the bounded probe through srun, then release the allocation.
```

`salloc` leaves the shell on the login node; `srun` executes on the allocated node. Always release an idle allocation. Merely using `salloc` or `srun` without the reservation does not select this pool. The reservation option is supported by upstream `salloc`. [BriCS advanced guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/), [Slurm salloc reference](https://slurm.schedmd.com/salloc.html)

A short unattended probe can use `sbatch --reservation=interactive probe.slurm`, **provided that the script contains no partition or conflicting QoS directive**. Existing `evals/personaplex-recovery.slurm`, `evals/moshi-rag-training-proof.slurm`, and `evals/qwen-delegation-probe.slurm` explicitly contain `#SBATCH --partition=workq`; simply appending the reservation option to those files is unsafe. Use a reviewed copy with that directive removed, preserve its resource requests and output paths, and unset any `SBATCH_PARTITION` or `SBATCH_QOS` environment override. Upstream Slurm accepts reservations in batch submission; environment options override script directives. This syntax does not waive BriCS's short-work limits. [Slurm sbatch reference](https://slurm.schedmd.com/sbatch.html)

## Queue visibility and moving pending probes

```bash
squeue --me --all
squeue --me --partition=interactive
scontrol show job JOBID
sacct -j JOBID --format=JobID,State,Partition,Elapsed,ExitCode
```

The hidden partition does not appear in ordinary `squeue --me`. Inspect the
submitted job's effective partition and reservation (`interactive`) and the
partition's policy (`interactive_qos`). The live job may retain the account's
job QoS `normal`; that is separate from the partition QoS applied by the site.
[BriCS advanced guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/),
[BriCS scheduling reference](https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/)

Slurm supports changing a job's `ReservationName`, but BriCS does not document whether updating an existing pending `workq` job reruns its automatic partition/QoS selection. **Do not assume `scontrol update JobId=... ReservationName=interactive` is safe.** For an urgent short probe, submit a correctly constructed replacement, verify its receipt and effective settings, then cancel the original pending probe while preventing duplicate execution. Preserve old/new IDs and dependency changes. Routine cancellation/resubmission loses queue age, so retain normal training jobs rather than repeatedly resubmitting them. [Slurm scontrol reference](https://slurm.schedmd.com/scontrol.html), [BriCS queue FAQ](https://docs.isambard.ac.uk/user-documentation/faqs/)

For longer checkpointed training, request realistic time and resources on `workq`; `--time-min` can help backfill only when the training process can save and resume within the smaller assigned walltime. [BriCS advanced guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm-advanced/)

## Live check — 14 October

Plain words: the reservation worked for this account. Committed wrapper
`evals/interactive-short-probe.slurm` selects the reservation without specifying
a partition or QoS. It directly executes one allowed committed probe payload;
the payload's SBATCH comments are ignored by bash. Code was pushed and pulled,
and its SHA256 matched before submission.

Replacement 7402644 was submitted at 11:00:45 UTC and was running on GH200
`nid010689` at the first subsequent check. `scontrol -a show partition interactive`
verified `QoS=interactive_qos`, `Hidden=YES`, `ReqResv=YES`, `PriorityTier=2`.
The job reported `Partition=interactive`, `Reservation=interactive`, `QOS=normal`.
Held original 7401863 was cancelled after the replacement started; no duplicate
compute ran. Repaired short PersonaPlex probe was then submitted as 7402673.

7402644 subsequently failed in the native whole-sequence forward: the model's
first-speaker prepend embedding was concatenated with integer token codes,
producing a 4096-versus-128 shape mismatch. This is a training integration
failure exposed by faster execution, not a scheduling failure. Do not drop the
prepend condition to claim training/inference parity. Prefix-aware training
still needs implementation and verification.

Slurm Submit/Start timestamps subsequently confirmed waits of 1, 2 and 0 seconds
for 7402644, 7402673 and 7402713 respectively. The third was a bounded Qwen
routing diagnostic that ran out of host memory during loading; its one-GPU
110 GiB retry is 7402749. These measured starts establish the benefit for these
probes, not a general wait-time guarantee or permission for bulk compute here.
