#!/usr/bin/env bash
# One scheduled allocation: bounded CPU staging, then the existing guarded reviewer.
set -euo pipefail
source_sha=${1:?exact source SHA required}
review_root=${2:?new review root required}
base=/projects/u6ada/duplex-live
test -n "${SLURM_JOB_ID:-}"
test "$(git rev-parse HEAD)" = "$source_sha"
case "$review_root" in "$base"/native-retention96-qwen-v1|"$base"/native-retention96-qwen-v2) ;; *) exit 2 ;; esac
mkdir "$review_root"
trap 'result=$?; /usr/bin/python3 -c '\''import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps(dict(actual_exit_code=int(sys.argv[2]),job_id=sys.argv[3],source=sys.argv[4],new_optimizer_updates=0))+"\n")'\'' "$review_root/bootstrap-terminal.json" "$result" "$SLURM_JOB_ID" "$source_sha"' EXIT
physical=$(readlink -f "$base")
worktree=$(pwd)
receipt="$base/reviewer-models/qwen3-32b-v1/download.json"
if test ! -e "$receipt"; then
  mkdir -p "$base/public-reviewer-cache-v1/home" "$base/public-reviewer-cache-v1/xet" "$base/public-reviewer-cache-v1/tmp"
  timeout --signal=TERM --kill-after=30s 1800s apptainer exec --cleanenv --no-home \
    --bind "$base:$base" --bind "$physical:$physical" \
    --env CUDA_VISIBLE_DEVICES= --env PYTHONDONTWRITEBYTECODE=1 \
    --env HF_HOME="$base/public-reviewer-cache-v1/home" \
    --env HF_XET_CACHE="$base/public-reviewer-cache-v1/xet" \
    --env TMPDIR="$base/public-reviewer-cache-v1/tmp" \
    --env HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
    --env HF_HUB_DISABLE_PROGRESS_BARS=1 --pwd "$worktree" \
    "$base/artifacts/native-v75/runtime/pytorch-new.sif" \
    "$base/envs/venus-arm64-v1/bin/python" -B -u -m data.venus.reviewer_model \
    --cache "$base/public-reviewer-cache-v1" --receipt "$receipt" \
    >"$review_root/download.log" 2>&1
fi
# The source module checks the full staged identity before loading any weights.
/usr/bin/python3 -B -u -m training.venus.isambard_job \
  --stage retention-qwen --source "$source_sha" \
  --worktree "$(basename "$worktree")" --root "$review_root"
