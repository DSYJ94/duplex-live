# Continue training after Isambard

This is a provider-neutral container recipe for the current Context Spanning
trainer. It is intended for NVIDIA GPU instances on AWS, GCP, Modal, Nebius or
Runpod. No provider resources are provisioned by this directory. The recipe
passes an amd64 image build and native Engine/training imports under local CPU
emulation. An amd64 GPU resume is still pending verification; do not treat this
recipe as a qualified deployment. The NVIDIA base is pinned to the tested digest.
The interrupted/uninterrupted optimizer/RNG tests also pass inside the amd64
image on CPU (two tests). Neither import checks nor CPU tests establish GPU
numerical equivalence or task quality.

Build from the repository root on the destination architecture:

```sh
docker build --platform linux/amd64 -t duplex-context:25.10 portable/context
```

Use a fresh image and wheels. Do not copy the ARM Isambard venv or Apptainer image
to an x86 instance. The base preserves the current NVIDIA Torch/CUDA stack;
native package versions were checked on Isambard. No weights, data, API keys or
SSH credentials enter the image through this Dockerfile.

Restore the artifact tree at `/projects/u6si/duplex-live` inside the container.
Existing approvals bind absolute paths and source commits. Preserve the git
history, pinned vendor source, model/tokenizer/codec files, voice provenance,
approved raw data and caches, captured histories, audit reports and checkpoints.
An adapter alone is insufficient. The local export now contains both full model
weights with read-back checksums, the original three independently revalidated
future components, and 25 additional synthetic captured histories (not approved
training labels). The isolated amd64 restore passes all three original data
audits. The real ARM checkpoint deserializes with finite optimizer/adapter state
and accepted CPU/Python/NumPy RNG on amd64; destination CUDA restore remains
untested. These checks cover the recorded recipe, not every future workload.

Recreate source checkouts from the saved Git bundle rather than copying linked
worktree `.git` files: those can point back to the old host. For each checkpoint's
required source commit, clone the bundle into `worktrees/recovery-COMMIT` and
check out that commit detached. Preserve the complete history for provenance
checks. Recreate the pinned vendor checkout with its Git metadata too. The
container bind supplies the original absolute artifact paths.

Before a GPU run, check imports and checkpoint-state tests using the **same source
commit as the checkpoint**. Set `SOURCE` to that restored immutable checkout:

```sh
docker run --rm --gpus '"device=0"' --ipc=host \
  -v "$RESTORED_TREE:/projects/u6si/duplex-live" \
  -w "$SOURCE" -e PYTHONPATH="$SOURCE" duplex-context:25.10 \
  -m unittest evals.harness.test_context_training_state
```

The trainer's `--resume-training .../training-state.pt` restores adapters, AdamW
moments, completed updates and Torch/Python/NumPy random states. It checks exact
data, recipe and source identity. Keep the parent `report.json`,
`reload-report.json`, data manifest and independent temporal audit beside it.
Use one visible GPU, matching the current trainer. A changed recipe or dataset
requires a distinct adaptation stage, not an exact-resume claim.

`bash portable/context/train.sh` provides the same audit/fit/resume path without
Slurm. It mounts the restored tree and runs as the host user. For example, after
the required destination GPU audits pass:

```sh
export DL_RESTORED_TREE=/mnt/duplex-live
export DL_TASK_COMMIT=ad827455fc5052fd7963b5bed4c274b94068a37c
export DL_RUN_ID=9990001  # choose a new unused numeric identifier
export CONTEXT_ACTUAL_HEAD_JOBS='7467981 7468102 7468611'
export CONTEXT_MODE=resume
export CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB=7475863
export CONTEXT_RESUME_TRAINING_JOB=7475992
export CONTEXT_ACTUAL_FIT_UPDATES=32
bash portable/context/train.sh
```

These are the recorded engineering-proof inputs, not a recommended final model.
Use `CONTEXT_MODE=audit` first for a new GPU and its new audit identifier in the
resume command. Keep the source commit fixed while resuming a saved state.

Destination acceptance requires strict base-weight coverage, independent data
audits, native stream/delay checks, finite forward/backward, saved-state reload
and free-running task/voice evaluation. CPU state tests have passed on Isambard;
cross-hardware bitwise numerical identity is not guaranteed. Current measured
training peak is about17.6GB for the tiny engineering proof, not a capacity
promise for a larger run or concurrent evaluation services.

For native evaluation, preserve the checkpoint's prefix/span convention and
attention mode. Current adapter loading restores the actual-history checkpoint's
native layout and MATH attention automatically. The historical `ad82745` source
requires explicit layout audit `runs/context-training-proof-7463825/history-audit.json`
and its recorded voice. Do not mix an unwrapped released-model trajectory into
the current actual-history trainer.

The exported directory has a `SHA256SUMS` list for its `tree` directory. On a
Linux destination, verify the transferred files before restoring source checkouts:

```sh
cd /mnt/cloud-export-20261016/tree
sha256sum --check ../SHA256SUMS
```

This establishes file integrity against the saved export. It does not approve
new captured trajectories for training or replace the destination GPU audits.


Plain words:the launcher now supports the later mixed-component recipes. Set
`CONTEXT_MIXED_TRAIN_DATA=1` for ordinary, private, control, lookup-prefix and
cancellation-acknowledgment components together. The default remains historical
early-only data. Mixed and explicit early-only modes are mutually exclusive.
For recent checkpoints, preserve `CONTEXT_ACTUAL_SAMPLING`,
`CONTEXT_ACTUAL_RETENTION_SAMPLING`, `CONTEXT_ACTUAL_RESPONSE_FIRST_WORD_WEIGHT`
and `CONTEXT_WARM_START_JOB` from that run's recipe and submission record. A resumed
stage still needs its original warm-start identity as well as
`CONTEXT_RESUME_TRAINING_JOB`; it restores the saved optimizer rather than
resetting it. Omitted or changed recipe settings must fail the checkpoint's
metadata check. New CLI options require a source commit that implements them.

Plain words:a local launcher integration check verifies mixed-data arguments,
resume/warm-start identity, objective/sampling settings, reload and rejection of
ambiguous data modes. Docker was mocked for that check: it does not establish a
new image build, GPU resume or quality result. The older amd64 CPU restore
result covers the three original components only. New17 October checkpoint,
raw-history and component files have separate read-back hash-verification
records in `runs/recovery-continuation-20261016`; regenerate the export-wide
checksum list after final artifact selection. Cloud validation remains paused.
