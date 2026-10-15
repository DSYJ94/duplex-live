#!/bin/bash
#SBATCH --job-name=dl-lychee-kernels
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-lychee-kernels-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
: "${LYCHEE_BUILD_JOB:?Successful isolated backend build required}"
[[ "$LYCHEE_BUILD_JOB" =~ ^[0-9]+$ ]]
BUILD="$DL/runs/lychee-vllm-build-$LYCHEE_BUILD_JOB"
RUNTIME="$DL/envs/lychee-vllm-${SLURM_JOB_ID}"
export DL BUILD RUNTIME
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
python - <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
build=Path(os.environ['BUILD']); record=json.loads((build/'build-report.json').read_text())
row,=record['wheels']; wheel=build/'wheels'/row['file']
if not record['built'] or hashlib.sha256(wheel.read_bytes()).hexdigest()!=row['sha256']:
 raise RuntimeError('Build/wheel receipt mismatch')
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',os.environ['RUNTIME'],str(wheel)],check=True)
PY
export PYTHONPATH="$RUNTIME:$DL/envs/lychee-probe-7415053:$DL/tmp/minicpm-recovery-7377673:$DL/repo"
python -u -m evals.lychee_vllm_preflight --runtime "$RUNTIME" --build "$BUILD" \
 --out "$DL/runs/lychee-vllm-preflight-${SLURM_JOB_ID}"
CONTAINER
