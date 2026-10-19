#!/bin/bash
#SBATCH --job-name=dl-native-pilot-development-session
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
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
[[ "${NATIVE_V2_CANDIDATE_JOB:-}" =~ ^[0-9]+$ ]]
test -z "${SLURM_ARRAY_JOB_ID:-}"
python3 - "$DL_SESSION_SOURCE" "$DL" "$NATIVE_V2_CANDIDATE_JOB" <<'PY'
import sys,json
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from evals.context_rollout_conditioning import verify_checkpoint
from evals.context_warm_ordinary_retention import IDS,DRIFT_SHA256
from evals.moshi_model_download import sha256
p=Path(sys.argv[2])/'runs'/('context-training-proof-'+sys.argv[3]);identity,r=verify_checkpoint(p)
w=r['warm_ordinary_retention'];audit=p/'warm-ordinary-reference-audit.json';a=json.loads(audit.read_text())
assert r['optimizer_steps'] in (8,40) and r['additional_native_v2']['complete24_per8']
assert w['mode']=='native520-approved-ordinary-nonpad-v1' and w['weight']==64
assert [x['id'] for x in w['examples']]==list(IDS) and w['complete53_diagnostic_sha256']==DRIFT_SHA256
assert sha256(audit)==r['warm_ordinary_reference_audit_sha256'] and a['policy']==w
assert sha256(p/'warm-ordinary-references.pt')==a['cache_sha256']
os_clock=r.get('warm_ordinary_clock_retention')
if os_clock:
 from evals.context_warm_ordinary_clock import CLOCK_IDS
 ca=p/'warm-ordinary-clock-reference-audit.json';cr=json.loads(ca.read_text())
 assert os_clock['mode']=='native520-approved-ordinary-clock-and-actual-v1' and os_clock['weight']==64
 assert os_clock['ids']==list(CLOCK_IDS) and cr['policy']==os_clock
 assert sha256(ca)==r['warm_ordinary_clock_reference_audit_sha256'] and cr['exact_self_checks']==14
 assert sha256(p/'warm-ordinary-clock-references.pt')==cr['cache_sha256']
print(json.dumps(dict(verified_candidate=identity,ordinary_anchor_profile=w['mode'])),flush=True)
PY
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
    candidate_job=os.environ['NATIVE_V2_CANDIDATE_JOB'],maximum_concurrent_gpus=2,training_approved=False,behavior_approved=False)
with (p/'execution.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
PY
exit "$STATUS"
