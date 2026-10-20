#!/bin/bash
#SBATCH --job-name=dl-native-audio49-watch
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=20:00:00
#SBATCH --output=logs/dl-native-audio49-watch-%j.out
set -euo pipefail
ulimit -c 0
: "${DL_WATCHER_SOURCE:?}" "${DL_WATCHER_COMMIT:?}" "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_AUDIO49_CHAIN_RECEIPT:?}"
test "$(git -C "$DL_WATCHER_SOURCE" rev-parse HEAD)" = "$DL_WATCHER_COMMIT"
test -z "$(git -C "$DL_WATCHER_SOURCE" status --porcelain)"
export DL_CHAIN_WATCHER_JOB="$SLURM_JOB_ID" PYTHONDONTWRITEBYTECODE=1
cd "$DL_WATCHER_SOURCE"
python3 - "$NATIVE_AUDIO49_CHAIN_RECEIPT" <<'PY'
import os,json,fcntl,tempfile,datetime
from pathlib import Path
p=Path(__import__('sys').argv[1])
with Path(str(p)+'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);r=json.loads(p.read_text())
 assert r['source']==os.environ['DL_TASK_COMMIT']
 r['scheduler_watcher']=dict(job=os.environ['SLURM_JOB_ID'],source=os.environ['DL_WATCHER_COMMIT'],started_utc=datetime.datetime.utcnow().isoformat()+'Z',survives_ssh_disconnect=True)
 r['server_watcher']['terminal']=True;r['server_watcher']['failure']='Detached login process disappeared after initial SSH session; no training submitted by it'
 r['maximum_submitted_active_jobs']=2;r['maximum_submitted_gpu_jobs']=1
 with tempfile.NamedTemporaryFile(mode='w',dir=str(p.parent),delete=False) as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno());name=f.name
 os.replace(name,str(p))
PY
exec python3 -u -m evals.native_audio49_chain watch --receipt "$NATIVE_AUDIO49_CHAIN_RECEIPT"
