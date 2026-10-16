# Continue training after Isambard

This is a provider-neutral container recipe for the current Context Spanning
trainer. It is intended for NVIDIA GPU instances on AWS, GCP, Modal, Nebius or
Runpod. No provider resources are provisioned by this directory. The image build
and an amd64 GPU resume are still pending verification; do not treat this recipe
as a qualified deployment.

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
An adapter alone is insufficient. Data export and dependency-closure verification
remain work in progress.

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
export DL_TASK_COMMIT=0e0e3634ea258761d2d9f16df6e3347ce9150834
export DL_RUN_ID=9990001  # choose a new unused numeric identifier
export CONTEXT_ACTUAL_HEAD_JOBS='7467981 7468102 7468611'
export CONTEXT_MODE=resume
export CONTEXT_ACTUAL_TEMPORAL_PREFLIGHT_JOB=7469530
export CONTEXT_RESUME_TRAINING_JOB=7474819
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
