#!/bin/bash
#SBATCH --job-name=dl-native-seen-retention-diagnostic
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dl-native-seen-retention-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable diagnostic source required}" "${DL_TASK_COMMIT:?}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
test -z "${SLURM_ARRAY_JOB_ID:-}"
cd "$DL_TASK_SOURCE"
python3 - "$DL" <<'PY'
import sys,json,os
from pathlib import Path
from evals.context_rollout_conditioning import verify_checkpoint
from evals.moshi_model_download import sha256
from evals.context_native_learning_curve import PINS,validate_lineage
b=Path(sys.argv[1]);parent,pr=verify_checkpoint(b/'runs/context-training-proof-7510331')
children=[verify_checkpoint(b/'runs'/('context-training-proof-'+p['job'])) for p in PINS]
validate_lineage(pr,children,parent['checkpoint_sha256'],pr['data_sha256'])
plan=json.load(open('evals/fixtures/native-seen-retention-plan-20261020.json'));assert len(plan['cases'])==12
from evals.native_seen_retention_diagnostic import case
assert plan['cases']==[case(i) for i in range(12)]
data=b/'runs/recovery-data-7407554';assert sha256(data/'ingestion.json')==plan['source_ingestion_sha256']
accepted={r['scenario_id']:r for r in json.load((data/'ingestion.json').open())['accepted']}
for binding in plan['source_bindings']:
 r=accepted[binding['scenario']];wav=Path(r['path'])
 assert r['split']=='train' and r['kind']=='simple'
 assert sha256(wav)==r['wav_sha256']==binding['wav_sha256']
 assert sha256(wav.with_suffix('.json'))==r['sidecar_sha256']==binding['sidecar_sha256']
print(json.dumps(dict(parent=parent,children=[x[0] for x in children],optimizer_steps=0,diagnostic_only=True)),flush=True)
PY
OUT="$DL/runs/native-seen-retention-session-$SLURM_JOB_ID"
mkdir "$OUT"
scontrol show job "$SLURM_JOB_ID" > "$OUT/allocation.txt"
run_case() {
 srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --gpus=1 --mem=64G \
  --time=00:08:00 --kill-on-bad-exit=1 --output="$DL/logs/dl-native-seen-retention-${SLURM_JOB_ID}_$1.out" \
  bash "$DL_TASK_SOURCE/evals/native-seen-retention-step.sh" "$1"
}
STATUS=0
for FIRST in 0 2 4 6 8 10; do
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
assert [r['index'] for r in rows]==list(range(12))
r=dict(cases=rows,all12_attempted=True,all_processes_succeeded=all(r['exit_code']==0 for r in rows),
 optimizer_steps=0,diagnostic_only=True,source=os.environ['DL_TASK_COMMIT'],checkpoint_jobs=['7510331','7556562','7558824'],
 text_temperature=.7,acoustic_temperature=.8,training_extension_or_adoption_approved=False)
with (p/'execution.json').open('x') as f:json.dump(r,f,indent=2)
PY
exit "$STATUS"
