#!/bin/bash
#SBATCH --job-name=dl-moshi-smoke
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/dl-moshi-smoke-%j.out
set -euo pipefail
: "${DATA_DIR:?Set DATA_DIR to a completed verified native conversion}"
: "${MODEL_MANIFEST:?Set MODEL_MANIFEST to the pinned download manifest}"
DL=/projects/u6si/duplex-live
cd "$DL/repo"
NATIVE_COMMIT=e6a55d2722a65870ef52a6c9f6ecfc0e90f38362
TRAINER_COMMIT=2acc879fe7c48f885a18f6cc9548bccb2674d87b
NATIVE="$DL/vendor/moshi-$NATIVE_COMMIT"
TRAINER="$DL/vendor/moshi-finetune-$TRAINER_COMMIT"
for spec in "moshi:$NATIVE_COMMIT:$NATIVE" "moshi-finetune:$TRAINER_COMMIT:$TRAINER"; do
    IFS=: read -r name revision target <<< "$spec"
    if [ ! -d "$target" ]; then
        git clone "https://github.com/kyutai-labs/$name.git" "$target"
        git -C "$target" checkout --detach "$revision"
    fi
    test "$(git -C "$target" rev-parse HEAD)" = "$revision"
    git -C "$target" diff --exit-code
done
ENV_DIR="$DL/envs/moshi-native-${SLURM_JOB_ID}"
OUT="$DL/runs/moshi-training-${SLURM_JOB_ID}"
ENV_OUT="$DL/runs/moshi-training-env-${SLURM_JOB_ID}"
BUILD_TMP="$DL/tmp/moshi-native-build-${SLURM_JOB_ID}"
mkdir -p "$BUILD_TMP"
export DL NATIVE TRAINER ENV_DIR OUT ENV_OUT DATA_DIR MODEL_MANIFEST
git rev-parse HEAD
apptainer exec --nv --bind /projects,/lus,/home --env PYTHONPATH="$DL/repo" \
    --env TMPDIR="$BUILD_TMP" --env HF_HOME="$DL/models/hf-cache" "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
python -m venv --system-site-packages "$ENV_DIR"
PY="$ENV_DIR/bin/python"
"$PY" -m pip install --ignore-installed --no-deps 'cmake==3.31.6'
export PATH="$ENV_DIR/bin:$PATH" CARGO_BUILD_JOBS=8
export CMAKE
CMAKE=$("$PY" -c 'import cmake; print(cmake.CMAKE_BIN_DIR + "/cmake")')
"$CMAKE" --version
"$PY" -m evals.moshi_environment_setup --requirements "$DL/repo/evals/moshi-training-requirements.txt" --out "$ENV_OUT"
export PYTHONPATH="$NATIVE/moshi:$TRAINER:$DL/repo" PYTHONDONTWRITEBYTECODE=1
"$PY" -c 'import sphn,torch,moshi; print("native-env", torch.__version__,torch.__file__,moshi.__file__); assert hasattr(sphn,"dataset_jsonl")'
"$PY" -m torch.distributed.run --standalone --nproc_per_node=1 \
    -m evals.moshi_training_smoke --trainer "$TRAINER" --data "$DATA_DIR" \
    --model-manifest "$MODEL_MANIFEST" --out "$OUT"
"$PY" -m evals.moshi_training_smoke --reload --trainer "$TRAINER" --data "$DATA_DIR" \
    --model-manifest "$MODEL_MANIFEST" --out "$OUT"
SCENARIO=$("$PY" -c 'import json,os; from pathlib import Path; d=json.loads((Path(os.environ["DATA_DIR"])/"ingestion.json").read_text()); print(d["accepted"][16]["scenario_id"])')
COMMON=(--model-manifest "$MODEL_MANIFEST" --batch "$DL/runs/synth-scale1-q/audio" --scenario "$SCENARIO" --seconds 40)
"$PY" -m evals.moshi_native_replay "${COMMON[@]}" --config "$OUT/base-config.json" \
    --out "$DL/runs/moshi-base-${SLURM_JOB_ID}"
"$PY" -m evals.moshi_native_replay "${COMMON[@]}" \
    --config "$OUT/train/checkpoints/checkpoint_000004/consolidated/config.json" \
    --adapter "$OUT/train/checkpoints/checkpoint_000004/consolidated/lora.safetensors" \
    --out "$DL/runs/moshi-adapter-${SLURM_JOB_ID}"
"$PY" -m pip freeze > "$ENV_OUT/pip-freeze.txt"
CONTAINER
