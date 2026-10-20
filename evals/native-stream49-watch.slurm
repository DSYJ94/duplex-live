#!/bin/bash
#SBATCH --job-name=dl-native-stream49-watch
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dl-native-stream49-watch-%j.out
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
 if r.get('server_watcher',{}).get('pid'):
  r['server_watcher']['terminal']=True;r['server_watcher']['failure']='Historical detached login process superseded by Slurm watcher'
 else:r['server_watcher']=dict(used=False,terminal=True)
 r['maximum_submitted_active_jobs']=2;r['maximum_submitted_gpu_jobs']=1
 with tempfile.NamedTemporaryFile(mode='w',dir=str(p.parent),delete=False) as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno());name=f.name
 os.replace(name,str(p))
PY
# The first readiness allocation remains HELD until this actual CPU suite passes.
export DL_CPU_TEST_REPORT="/projects/u6si/duplex-live/runs/native-stream49-cpu-preflight-$SLURM_JOB_ID.json"
export DL_CPU_EXPECTED_TESTS=286
if ! bash "$DL_TASK_SOURCE/evals/native-v2-tests.slurm"; then
 python3 - "$NATIVE_AUDIO49_CHAIN_RECEIPT" <<'FAIL'
import json,sys,subprocess,os
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(p.read_text());j=r['jobs']['readiness']
assert r['source']==os.environ['DL_TASK_COMMIT']
assert r['submission_attempts']['readiness']['status']=='held'
subprocess.check_call(['scancel',j])
r['watcher_terminal']=dict(state='FAILED',reason='CPU suite failed before readiness release',no_training=True)
p.write_text(json.dumps(r,indent=2)+'\n')
FAIL
 exit 1
fi
python3 - "$NATIVE_AUDIO49_CHAIN_RECEIPT" <<'RELEASE'
import json,sys,os,hashlib,subprocess,fcntl,tempfile
from pathlib import Path
p=Path(sys.argv[1])
with Path(str(p)+'.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX);r=json.loads(p.read_text());j=r['jobs']['readiness']
 report=Path(os.environ['DL_CPU_TEST_REPORT']);v=json.loads(report.read_text())
 assert v['passed'] and v['tests_run']==286 and v['source']==r['source']==os.environ['DL_TASK_COMMIT'] and v['allocation_job']==os.environ['SLURM_JOB_ID'] and not v['cuda_initialized']
 assert r['native_stream_auxiliary'] and r['submission_attempts']['readiness']['status']=='held'
 settings=subprocess.check_output(['scontrol','show','job',j],universal_newlines=True)
 assert 'JobState=PENDING' in settings and 'Reason=JobHeldUser' in settings and 'TimeLimit=02:00:00' in settings and 'Reservation=interactive' in settings and 'TresPerNode=gres/gpu:1' in settings
 active=subprocess.check_output(['squeue','--all','-u',os.environ['USER'],'-h','-o','%i'],universal_newlines=True).split()
 assert set(active)=={j,os.environ['SLURM_JOB_ID']}
 r['cpu_preflight']=dict(path=str(report),sha256=hashlib.sha256(report.read_bytes()).hexdigest(),tests=286)
 # Record release intent before the one non-repeatable scheduler mutation.
 r['submission_attempts']['readiness']['status']='releasing'
 def save():
  with tempfile.NamedTemporaryFile(mode='w',dir=str(p.parent),delete=False) as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno());name=f.name
  os.replace(name,str(p))
 save();subprocess.check_call(['scontrol','release',j])
 r['submission_attempts']['readiness']['status']='released';r.setdefault('released',[]).append(j);save()
RELEASE
unset DL_CPU_TEST_REPORT DL_CPU_EXPECTED_TESTS
exec python3 -u -m evals.native_audio49_chain watch --receipt "$NATIVE_AUDIO49_CHAIN_RECEIPT"
