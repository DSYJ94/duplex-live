#!/bin/bash
#SBATCH --job-name=dl-native-precision-diagnostic
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-precision-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable diagnostic source required}" "${DL_TASK_COMMIT:?}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
test -z "${SLURM_ARRAY_JOB_ID:-}"
cd "$DL_TASK_SOURCE"
# Full existing suite plus locked precision policy tests before any GPU inference.
bash evals/native-v2-tests.slurm
python3 - <<'PY2'
from pathlib import Path
import json
from evals.context_rollout_conditioning import verify_checkpoint
b=Path('/projects/u6si/duplex-live');parent,p=verify_checkpoint(b/'runs/context-training-proof-7510331');child,c=verify_checkpoint(b/'runs/context-training-proof-7585424')
assert p['optimizer_steps']==520 and c['optimizer_steps']==8 and c['schema']=='dl-context-actual-audio49-adapter-proof-v1'
assert p['voice_sha256']==c['voice_sha256'] and p['data_sha256']==c['data_sha256']
print(json.dumps(dict(parent=parent,child=child,optimizer_steps=0,training_approved=False)),flush=True)
PY2
OUT="$DL/runs/native-precision-session-$SLURM_JOB_ID"
mkdir "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
run_case() {
 srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --gpus=1 --mem=64G \
  --time=00:08:00 --kill-on-bad-exit=1 --output="$DL/logs/dl-native-precision-${SLURM_JOB_ID}_$1.out" \
  bash "$DL_TASK_SOURCE/evals/native-precision-step.sh" "$1"
}
STATUS=0
for FIRST in 0 2 4 6 8 10 12 14; do
 run_case "$FIRST" & LEFT=$!
 run_case "$((FIRST+1))" & RIGHT=$!
 RC=0; wait "$LEFT" || RC=$?; printf '%s\t%s\n' "$FIRST" "$RC" >> "$OUT/case-exit-codes.tsv"
 if [ "$RC" -ne 0 ]; then STATUS=1; fi
 RC=0; wait "$RIGHT" || RC=$?; printf '%s\t%s\n' "$((FIRST+1))" "$RC" >> "$OUT/case-exit-codes.tsv"
 if [ "$RC" -ne 0 ]; then STATUS=1; fi
done
python3 - "$OUT" <<'PY'
import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);rows=[dict(index=int(i),exit_code=int(code)) for i,code in (s.split() for s in (p/'case-exit-codes.tsv').read_text().splitlines())]
assert [r['index'] for r in rows]==list(range(16))
r=dict(cases=rows,all16_attempted=True,all_processes_succeeded=all(r['exit_code']==0 for r in rows),
 optimizer_steps=0,diagnostic_only=True,source=os.environ['DL_TASK_COMMIT'],
 text_temperature=.7,acoustic_temperature=.8,training_extension_or_adoption_approved=False)
with (p/'execution.json').open('x') as f:json.dump(r,f,indent=2)
PY
exit "$STATUS"
