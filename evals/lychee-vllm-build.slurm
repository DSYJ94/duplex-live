#!/bin/bash
#SBATCH --job-name=dl-lychee-vllm
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-lychee-vllm-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
SOURCE="$DL/vendor/lychee-fd-7bb1068c59b578e1f8e65fb762cd26708c14cc6d"
COMMIT=7bb1068c59b578e1f8e65fb762cd26708c14cc6d
OUT="$DL/runs/lychee-vllm-build-${SLURM_JOB_ID}"
BUILD="$DL/tmp/lychee-vllm-source-${SLURM_JOB_ID}"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
mkdir "$OUT"
mkdir "$OUT/tmp"
export TMPDIR="$OUT/tmp"
git init "$BUILD"
git -C "$BUILD" remote add origin https://github.com/HITsz-TMG/Lychee-FD.git
git -C "$BUILD" fetch --depth=1 --filter=blob:none origin "$COMMIT"
git -C "$BUILD" sparse-checkout set third_party/vllm
git -C "$BUILD" checkout --detach "$COMMIT"
test -f "$BUILD/third_party/vllm/setup.py"
test -f "$BUILD/third_party/vllm/pyproject.toml"
export DL OUT BUILD
apptainer exec --nv --bind /projects,/lus,/home \
 --env TMPDIR="$OUT/tmp" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
python - <<'PY'
import torch, platform, json, os, subprocess
from pathlib import Path
x=torch.randn(16,16,device='cuda'); (x@x).sum().item()
record={'machine':platform.machine(),'torch':torch.__version__,'cuda':torch.version.cuda,
 'gpu':torch.cuda.get_device_name(),'scope':'isolated patched vLLM native build, no model/task acceptance',
 'source_commit':'7bb1068c59b578e1f8e65fb762cd26708c14cc6d',
 'repo_commit':subprocess.check_output(['git','-C',os.environ['DL']+'/repo','rev-parse','HEAD'],text=True).strip()}
Path(os.environ['OUT']+'/environment.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
python -m pip install --no-deps --target "$OUT/build-deps" \
 cmake==3.31.6 ninja==1.11.1.3 setuptools-scm==8.3.1
export PYTHONPATH="$OUT/build-deps:$DL/repo"
export PATH="$OUT/build-deps/bin:$PATH"
export MAX_JOBS=8 NVCC_THREADS=1 TORCH_CUDA_ARCH_LIST=9.0 VLLM_TARGET_DEVICE=cuda
export SETUPTOOLS_SCM_PRETEND_VERSION=0.6.5+lychee
export CMAKE_BUILD_PARALLEL_LEVEL=8
export VLLM_FETCHCONTENT_BASE_DIR="$OUT/fetch-content"
python -m evals.lychee_cuda13_compat --source "$BUILD" --out "$OUT/compatibility.json"
python -m pip wheel --no-build-isolation --no-deps \
 --wheel-dir "$OUT/wheels" "$BUILD/third_party/vllm"
python - <<'PY'
import hashlib,json,os
from pathlib import Path
out=Path(os.environ['OUT']); rows=[]
for p in (out/'wheels').glob('*.whl'):
 rows.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
if len(rows)!=1: raise RuntimeError('Expected one locally built patched vLLM wheel')
(out/'build-report.json').write_text(json.dumps({'built':True,'wheels':rows,
 'torch_replaced':False,'runtime_or_model_load_tested':False},indent=2)+'\n')
print(json.dumps(rows),flush=True)
PY
CONTAINER
