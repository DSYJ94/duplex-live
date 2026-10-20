#!/bin/bash
#SBATCH --job-name=dl-native-fresh2-session
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-native-fresh2-session-%j.out
set -euo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
DL=/projects/u6si/duplex-live
test "${NATIVE_FRESH_SOURCE_PROFILE:-owner-approved-turbo-spelling-v1}" = owner-approved-turbo-spelling-v1
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}"
: "${NATIVE_FRESH_INPUTS_JOB:?}" "${NATIVE_FRESH_QUALITY_JOB:?}"
: "${NATIVE_FRESH_CANDIDATE_JOB:?}" "${NATIVE_FRESH_BEHAVIOR_REVIEW:?}"
for NAME in NATIVE_FRESH_INPUTS_JOB NATIVE_FRESH_QUALITY_JOB NATIVE_FRESH_CANDIDATE_JOB; do [[ "${!NAME}" =~ ^[0-9]+$ ]]; done
test -z "${SLURM_ARRAY_JOB_ID:-}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
bash evals/native-v2-tests.slurm
OUT="$DL/runs/native-fresh2-session-$SLURM_JOB_ID"
mkdir "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
apptainer exec --bind /projects,/lus,/home --env PYTHONPATH="$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OPENAI_API_KEY= "$DL/pytorch-new.sif" python -u \
 -m evals.native_fresh_session preflight --candidate "$NATIVE_FRESH_CANDIDATE_JOB" \
 --review "$NATIVE_FRESH_BEHAVIOR_REVIEW" --data "$DL/runs/native-fresh2-inputs-$NATIVE_FRESH_INPUTS_JOB" \
 --quality "$DL/runs/native-fresh2-source-reapproval-$NATIVE_FRESH_QUALITY_JOB/quality.json" \
 --out "$OUT/preflight.json"
python3 - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads((p/'preflight.json').read_text());assert r['complete40'] and r['no_passing_subset']
with (p/'batches.tsv').open('x') as f:
 for batch in r['batches']:
  n=r['cases'][batch[0]]['required_gpus'];assert sum(r['cases'][i]['required_gpus'] for i in batch)<=2
  f.write('%d\t%d\t%s\n'%(n,batch[0],str(batch[1]) if len(batch)==2 else '-'))
PY
run_case() {
 local INDEX="$1" GPUS="$2" CPUS=8 MEMORY=64G
 if [ "$GPUS" = 2 ]; then CPUS=16; MEMORY=128G; fi
 srun --exclusive --exact --ntasks=1 --cpus-per-task="$CPUS" --gpus="$GPUS" --mem="$MEMORY" \
  --time=00:12:00 --kill-on-bad-exit=1 --output="$DL/logs/dl-native-fresh2-evaluate-${SLURM_JOB_ID}_$INDEX.out" \
  bash "$DL_TASK_SOURCE/evals/native-fresh2-session-step.sh" "$INDEX"
}
STATUS=0
while IFS=$'\t' read -r GPUS LEFT RIGHT; do
 run_case "$LEFT" "$GPUS" & LPID=$!
 if [ "$RIGHT" != - ]; then run_case "$RIGHT" "$GPUS" & RPID=$!; fi
 RC=0; wait "$LPID" || RC=$?; printf '%s\t%s\n' "$LEFT" "$RC" >> "$OUT/case-exit-codes.tsv"
 if [ "$RC" -ne 0 ]; then STATUS=1; fi
 if [ "$RIGHT" != - ]; then
  RC=0; wait "$RPID" || RC=$?; printf '%s\t%s\n' "$RIGHT" "$RC" >> "$OUT/case-exit-codes.tsv"
  if [ "$RC" -ne 0 ]; then STATUS=1; fi
 fi
done < "$OUT/batches.tsv"
python3 - "$OUT" <<'PY'
import json,os,sys
from pathlib import Path
p=Path(sys.argv[1]);rows=[dict(index=int(i),exit_code=int(c)) for i,c in (line.split() for line in (p/'case-exit-codes.tsv').read_text().splitlines())]
assert sorted(r['index'] for r in rows)==list(range(40))
r=dict(source=os.environ['DL_TASK_COMMIT'],scheduler_array=False,all40_attempted=True,cases=rows,
 all_processes_succeeded=all(r['exit_code']==0 for r in rows),evaluation_only=True,
 optimizer_updates=0,training_approved=False,checkpoint_adopted=False,full20_task_qualification_passed=False)
with (p/'execution.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
PY
exit "$STATUS"
