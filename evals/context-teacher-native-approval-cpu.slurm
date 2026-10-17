#!/bin/bash
#SBATCH --job-name=dl-native-teacher-cpu
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-native-teacher-cpu-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${CONTEXT_TEACHER_INPUT_CAPTURE_JOBS:?Native teacher traces required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
run() {
 apptainer exec --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 "$DL/pytorch-new.sif" python -u "$@"
}
run -c 'import sys,json,unittest; sys.path.insert(0,json.load(open("/projects/u6si/duplex-live/runs/context-spanning-assets-7454205/manifest.json"))["source"]); unittest.main(module="evals.harness.test_context_teacher_native_approval")'
CAPTURES=()
for J in $CONTEXT_TEACHER_INPUT_CAPTURE_JOBS; do
 [[ "$J" =~ ^[0-9]+$ ]]
 CAPTURES+=(--capture "$DL/runs/context-training-proof-$J")
done
run -m evals.context_teacher_native_approval \
 --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --tokenizer "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/tokenizer_spm_32k_3.model" \
 --voice "$DL/runs/context-training-proof-7505957/voice.pt" \
 --scope "$DL/runs/context-retention-teacher-audit-7507449/report.json" \
 --cache "$DL/runs/recovery-mixed-codes-7411513" --cache "$DL/runs/recovery-codes-7434926" \
 --cache "$DL/runs/recovery-codes-7435538" "${CAPTURES[@]}" \
 --out "$DL/runs/context-teacher-native-approval-${SLURM_JOB_ID}"
