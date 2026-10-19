#!/bin/bash
set -euo pipefail
ulimit -c 0
INDEX="${1:?Development case index required}"
[[ "$INDEX" =~ ^[0-9]+$ && "$INDEX" -lt 10 ]]
test "${DL_TASK_COMMIT:-}" = cf28726b44ed0621dde6bfd27257709574f5e9ed
[[ "${NATIVE_V2_CANDIDATE_JOB:-}" =~ ^[0-9]+$ ]]
python3 - "$INDEX" <<'PY'
import os,json,sys,datetime
from pathlib import Path
i=int(sys.argv[1]);devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
assert len(devices)==len(set(devices))==(1 if i<6 else 2)
assert os.environ['SLURM_STEP_ID'].isdigit()
r=dict(schema='dl-native-pilot-development-step-v1',started_utc=datetime.datetime.utcnow().isoformat()+'Z',
    actual_allocation_job=os.environ['SLURM_JOB_ID'],actual_step_id=os.environ['SLURM_STEP_ID'],
    scheduler_array=False,legacy_artifact_case_group=os.environ['SLURM_JOB_ID'],case_index=i,
    assigned_cuda_devices=devices,payload_source=os.environ['DL_TASK_COMMIT'],
    session_source=os.environ['DL_SESSION_COMMIT'],candidate_job=os.environ['NATIVE_V2_CANDIDATE_JOB'])
p=Path('/projects/u6si/duplex-live/runs')/('native-pilot-session-'+os.environ['SLURM_JOB_ID'])
with (p/('case-%d-launch.json'%i)).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
PY
# Compatibility ONLY for the original payload's case filenames and routing.
# These labels do not claim this single allocation is a scheduler array; actual
# allocation/step identities and GPU assignments are recorded independently above.
export SLURM_ARRAY_JOB_ID="$SLURM_JOB_ID" SLURM_ARRAY_TASK_ID="$INDEX"
exec bash "$DL_TASK_SOURCE/evals/native-v2-development.slurm"
