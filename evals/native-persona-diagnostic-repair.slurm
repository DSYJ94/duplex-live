#!/bin/bash
#SBATCH --job-name=dl-native-persona-repair
#SBATCH --reservation=interactive
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:10:00
#SBATCH --output=logs/dl-native-persona-repair-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
python3 - <<'PY'
import json
from pathlib import Path
from evals.moshi_model_download import sha256
b=Path('/projects/u6si/duplex-live/runs');execution=json.load(open(b/'native-persona-diagnostic-session-7570828/execution.json'))
assert execution['complete10_attempted'] and execution['payload_source']=='36d3c2d7eddc3a7c19aff828990623e0a412a3d5'
assert [r['index'] for r in execution['cases'] if r['exit_code']!=0]==[4,5]
for i in range(10):
 p=b/('native-persona-diagnostic-7570828_'+str(i))
 if i in (4,5):assert not (p/'report.json').exists()
 else:
  r=json.load(open(p/'report.json'));assert sha256(p/'model.wav')==r['audio_sha256']
  assert r['loading']['native_sampling_diagnostic']['diagnostic_outputs_must_not_be_trained_on']
PY
OUT="$DL/runs/native-persona-diagnostic-repair-$SLURM_JOB_ID"
mkdir "$OUT"
run_case() {
 srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --gpus=1 --mem=64G --time=00:08:00 --kill-on-bad-exit=1 \
 --output="$DL/logs/dl-native-persona-diagnostic-${SLURM_JOB_ID}_$1.out" bash "$DL_TASK_SOURCE/evals/native-persona-diagnostic-step.sh" "$1"
}
run_case 4 & LEFT=$!
run_case 5 & RIGHT=$!
STATUS=0
wait "$LEFT" || STATUS=1
wait "$RIGHT" || STATUS=1
python3 - "$OUT" "$STATUS" <<'PY'
import json,sys,os
from pathlib import Path
p=Path(sys.argv[1]);job=os.environ['SLURM_JOB_ID']
r=dict(schema='dl-native-persona-diagnostic-invocation-repair-v1',original_session='7570828',repair_session=job,
    repair_indices=[4,5],unchanged_original_indices=[0,1,2,3,6,7,8,9],all_processes_succeeded=sys.argv[2]=='0',
    source=os.environ['DL_TASK_COMMIT'],diagnostic_only=True,training_approved=False,behavior_approved=False,
    cases=[dict(index=i,run='/projects/u6si/duplex-live/runs/native-persona-diagnostic-'+(job if i in (4,5) else '7570828')+'_'+str(i)) for i in range(10)])
(p/'execution.json').write_text(json.dumps(r,indent=2)+'\n')
PY
exit "$STATUS"
