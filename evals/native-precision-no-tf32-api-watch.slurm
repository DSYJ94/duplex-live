#!/bin/bash
#SBATCH --job-name=dl-native-precision-tf32-api-watch
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-native-precision-tf32-api-watch-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?}" "${DL_TASK_COMMIT:?}" "${NATIVE_PRECISION_CHAIN_RECEIPT:?}"
cd "$DL_TASK_SOURCE"
test "$(git rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git status --porcelain)"
export PYTHONDONTWRITEBYTECODE=1 DL_CHAIN_WATCHER_JOB="$SLURM_JOB_ID"
PREFLIGHT="$DL/runs/native-precision-tf32-api-preflight-$SLURM_JOB_ID.json"
apptainer exec --bind /projects,/lus,/home --env PYTHONPATH="$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=1 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u - "$PREFLIGHT" <<'PY'
import json,os,sys,unittest,torch
suite=unittest.defaultTestLoader.loadTestsFromName('evals.harness.test_native_precision_diagnostic')
result=unittest.TextTestRunner(verbosity=2).run(suite)
passed=result.wasSuccessful() and result.testsRun==7 and not torch.cuda.is_initialized()
r=dict(schema='dl-native-tf32-cpu-api-preflight-v1',source=os.environ['DL_TASK_COMMIT'],
 allocation_job=os.environ['SLURM_JOB_ID'],
 passed=passed,tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),
 torch=str(torch.__version__),cuda_initialized=torch.cuda.is_initialized(),
 cpu_backend_contract_only=True,no_native_inference=True,optimizer_updates=0)
with open(sys.argv[1],'x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
raise SystemExit(0 if passed else 1)
PY
python3 - "$NATIVE_PRECISION_CHAIN_RECEIPT" "$PREFLIGHT" <<'PY'
import fcntl,json,sys
from pathlib import Path
from evals.native_audio49_chain import digest,save
p=Path(sys.argv[1]);pre=Path(sys.argv[2])
with Path(str(p)+'.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX);r=json.loads(p.read_text());d=json.loads(pre.read_text())
 assert r['cpu_api_preflight_required'] and d['passed'] and d['source']==r['source'] and d['tests_run']==7 and not d['cuda_initialized']
 r['cpu_api_preflight']=dict(path=str(pre),sha256=digest(pre));save(p,r)
PY
python3 -u -m evals.native_precision_chain --receipt "$NATIVE_PRECISION_CHAIN_RECEIPT"
