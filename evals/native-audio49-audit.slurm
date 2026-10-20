#!/bin/bash
#SBATCH --job-name=dl-native-audio49-audit
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-audio49-audit-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_AUDIO49_PREFLIGHT_JOB:?}"
[[ "$NATIVE_AUDIO49_PREFLIGHT_JOB" =~ ^[0-9]+$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
MOSHI="$DL/vendor/moshi-rag-8c6dfc101b7871baa428424bcdc583b74fb561d9"
test "$(git -C "$MOSHI" rev-parse HEAD)" = 8c6dfc101b7871baa428424bcdc583b74fb561d9
cd "$DL_TASK_SOURCE"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$MOSHI/moshi:$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env DL_TASK_COMMIT="$DL_TASK_COMMIT" --env HF_HOME="$DL/models/hf-cache" \
 --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -m evals.context_native_audio49_audit --phase audit \
 --preflight "$DL/runs/native-audio49-preflight-$NATIVE_AUDIO49_PREFLIGHT_JOB" \
 --out "$DL/runs/native-audio49-audit-$SLURM_JOB_ID"
