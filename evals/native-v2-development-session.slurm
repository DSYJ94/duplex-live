#!/bin/bash
#SBATCH --job-name=dl-native-pilot-development-session
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:15:00
#SBATCH --output=logs/dl-native-pilot-session-%j.out
# One finite exploratory pilot check; no partition/QoS override or actual array.
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_SESSION_SOURCE:?Immutable session wrapper required}"
: "${DL_SESSION_COMMIT:?Immutable session wrapper identity required}"
: "${DL_TASK_SOURCE:?Original immutable native payload required}"
test "$(git -C "$DL_SESSION_SOURCE" rev-parse HEAD)" = "$DL_SESSION_COMMIT"
test -z "$(git -C "$DL_SESSION_SOURCE" status --porcelain)"
test "${DL_TASK_COMMIT:-}" = cf28726b44ed0621dde6bfd27257709574f5e9ed
test "${NATIVE_V2_CANDIDATE_JOB:-}" = 7549991
test -z "${SLURM_ARRAY_JOB_ID:-}"
OUT="$DL/runs/native-pilot-session-$SLURM_JOB_ID"
mkdir "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
run_case() {
 srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --gpus="$2" --mem="$3" \
  --time=00:10:00 --kill-on-bad-exit=1 \
  --output="$DL/logs/dl-native-v2-development-${SLURM_JOB_ID}_$1.out" \
  bash "$DL_SESSION_SOURCE/evals/native-v2-development-session-step.sh" "$1"
}
STATUS=0
for FIRST in 0 2 4; do
 run_case "$FIRST" 1 64G & LEFT=$!
 run_case "$((FIRST+1))" 1 64G & RIGHT=$!
 RC=0; wait "$LEFT" || RC=$?; printf '%s\t%s\n' "$FIRST" "$RC" >> "$OUT/case-exit-codes.tsv"
 if [ "$RC" -ne 0 ]; then STATUS=1; fi
 RC=0; wait "$RIGHT" || RC=$?; printf '%s\t%s\n' "$((FIRST+1))" "$RC" >> "$OUT/case-exit-codes.tsv"
 if [ "$RC" -ne 0 ]; then STATUS=1; fi
done
for INDEX in 6 7 8 9; do
 RC=0; run_case "$INDEX" 2 96G || RC=$?
 printf '%s\t%s\n' "$INDEX" "$RC" >> "$OUT/case-exit-codes.tsv"
 if [ "$RC" -ne 0 ]; then STATUS=1; fi
done
python3 - "$OUT" <<'PY'
import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);rows=[dict(index=int(a),exit_code=int(b)) for a,b in
    (line.split() for line in (p/'case-exit-codes.tsv').read_text().splitlines())]
assert [r['index'] for r in rows]==list(range(10))
r=dict(schema='dl-native-pilot-development-session-v1',actual_allocation_job=os.environ['SLURM_JOB_ID'],
    scheduler_array=False,legacy_artifact_case_group=os.environ['SLURM_JOB_ID'],cases=rows,
    complete10_attempted=True,all_processes_succeeded=all(x['exit_code']==0 for x in rows),
    payload_source=os.environ['DL_TASK_COMMIT'],session_source=os.environ['DL_SESSION_COMMIT'],
    candidate_job='7549991',maximum_concurrent_gpus=2,training_approved=False,behavior_approved=False)
with (p/'execution.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
PY
exit "$STATUS"
