#!/bin/bash
#SBATCH --job-name=dl-native-persona-diagnostic
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-persona-diagnostic-session-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable diagnostic source required}" "${DL_TASK_COMMIT:?}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
test "${NATIVE_V2_CANDIDATE_JOB:-}" = 7558824
test "${DL_SESSION_COMMIT:-}" = "$DL_TASK_COMMIT"
test -z "${SLURM_ARRAY_JOB_ID:-}"
cd "$DL_TASK_SOURCE"
python3 - "$DL" <<'CHECK'
import json,sys,hashlib
from pathlib import Path
from evals.context_rollout_conditioning import verify_checkpoint
from evals.native_persona_diagnostic import case,scoped_persona
b=Path(sys.argv[1]);i,r=verify_checkpoint(b/'runs/context-training-proof-7558824')
assert r['optimizer_steps']==40 and i['checkpoint_sha256']=='b4361124025b339c02e354b7729699a520da74400927eb1aaa9a5029136264b6'
p=json.load(open('evals/fixtures/native-persona-diagnostic-plan-20261020.json'))
assert p['cases']==[case(i) for i in range(10)] and p['optimizer_updates']==0
assert hashlib.sha256(scoped_persona().encode()).hexdigest()==p['diagnostic_persona_sha256']
print(json.dumps(dict(verified_candidate=i,diagnostic_only=True,optimizer_updates=0)),flush=True)
CHECK
OUT="$DL/runs/native-persona-diagnostic-session-$SLURM_JOB_ID"
mkdir "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
run_case() {
 srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --gpus="$2" --mem="$3" \
  --time=00:10:00 --kill-on-bad-exit=1 \
  --output="$DL/logs/dl-native-persona-diagnostic-${SLURM_JOB_ID}_$1.out" \
  bash "$DL_TASK_SOURCE/evals/native-persona-diagnostic-step.sh" "$1"
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
r=dict(schema='dl-native-persona-diagnostic-session-v1',actual_allocation_job=os.environ['SLURM_JOB_ID'],
    scheduler_array=False,legacy_artifact_case_group=os.environ['SLURM_JOB_ID'],cases=rows,
    complete10_attempted=True,all_processes_succeeded=all(x['exit_code']==0 for x in rows),
    payload_source=os.environ['DL_TASK_COMMIT'],session_source=os.environ['DL_SESSION_COMMIT'],
    candidate_job=os.environ['NATIVE_V2_CANDIDATE_JOB'],maximum_concurrent_gpus=2,training_approved=False,behavior_approved=False)
with (p/'execution.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
PY
exit "$STATUS"
