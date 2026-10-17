#!/bin/bash
#SBATCH --job-name=dl-bayling-runtime
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-bayling-runtime-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
VENDOR=$DL/vendor/bayling-duplex-41669ac04bbcbfaf5b1be3a2cc563e4b3752d041
test "$(git -C "$VENDOR" rev-parse HEAD)" = 41669ac04bbcbfaf5b1be3a2cc563e4b3752d041
test -z "$(git -C "$VENDOR" status --porcelain)"
OVERLAY=$DL/tmp/bayling-runtime-$SLURM_JOB_ID
OUT=$DL/runs/bayling-runtime-$SLURM_JOB_ID
mkdir "$OVERLAY" "$OUT"
cd "$DL_TASK_SOURCE"
# Preserve the working NGC Torch/driver stack. Install only named isolated
# Python dependencies; missing transitive dependencies must fail the import.
apptainer exec --bind /projects,/lus,/home "$DL/pytorch-new.sif" \
 python -m pip install --no-deps --disable-pip-version-check --target "$OVERLAY" \
 -r "$DL_TASK_SOURCE/evals/assets/bayling-runtime-requirements.txt"
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$OVERLAY:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$VENDOR:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=4 \
 --env DL_BAYLING_OVERLAY="$OVERLAY" --env DL_BAYLING_OUT="$OUT" \
 "$DL/pytorch-new.sif" python - <<'PY'
import json,os
from pathlib import Path
import torch,transformers,tokenizers,diffusers,hyperpyyaml,tiktoken,conformer,soundfile,scipy
from bayling_duplex import BayLingDuplex
from evals.moshi_model_download import sha256
assert torch.__version__.startswith('2.9.0') and transformers.__version__=='4.46.3'
record={'schema':'dl-bayling-runtime-v1','overlay':os.environ['DL_BAYLING_OVERLAY'],
    'requirements_sha256':sha256(Path('evals/assets/bayling-runtime-requirements.txt')),
    'imports_passed':True,'torch':torch.__version__,'torch_path':torch.__file__,
    'transformers':transformers.__version__,'transformers_path':transformers.__file__,
    'tokenizers':tokenizers.__version__,'diffusers':diffusers.__version__,
    'gpu_execution_tested':False,'quality_qualified':False}
(Path(os.environ['DL_BAYLING_OUT'])/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
PY
