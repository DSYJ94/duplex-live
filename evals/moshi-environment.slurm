#!/bin/bash
#SBATCH --job-name=dl-moshi-env
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --output=logs/dl-moshi-env-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
OUT="$DL/runs/moshi-environment-${SLURM_JOB_ID}"
ENV_DIR="$DL/envs/moshi-rag-${SLURM_JOB_ID}"
MOSHI_COMMIT=8c6dfc101b7871baa428424bcdc583b74fb561d9
XFORMERS_COMMIT=41531cee4ca4e21f703968b31ffabda3d028f8fc
SOURCE="$DL/vendor/moshi-rag-$MOSHI_COMMIT"
ATTENTION="$DL/vendor/xformers-$XFORMERS_COMMIT"
BUILD_TMP="$DL/tmp/moshi-build-${SLURM_JOB_ID}"
mkdir -p "$DL/vendor" "$DL/envs"
mkdir -p "$BUILD_TMP"
mkdir "$OUT"
if [ ! -d "$SOURCE" ]; then
    git clone https://github.com/kyutai-labs/moshi-rag.git "$SOURCE"
    git -C "$SOURCE" checkout --detach "$MOSHI_COMMIT"
fi
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$MOSHI_COMMIT"
test -z "$(git -C "$SOURCE" status --porcelain)"
if [ ! -d "$ATTENTION" ]; then
    git clone --depth 1 --branch v0.0.33.post2 https://github.com/facebookresearch/xformers.git "$ATTENTION"
    git -C "$ATTENTION" submodule update --init --depth 1 third_party/cutlass
fi
# Packaging reads flash_attn metadata even when its GPU kernels are disabled.
git -C "$ATTENTION" submodule update --init --depth 1 third_party/flash-attention
test "$(git -C "$ATTENTION" rev-parse HEAD)" = "$XFORMERS_COMMIT"
git rev-parse HEAD
export DL OUT ENV_DIR SOURCE ATTENTION
apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH= --env TMPDIR="$BUILD_TMP" "$DL/pytorch-new.sif" bash -eu <<'CONTAINER'
python -m venv --system-site-packages "$ENV_DIR"
PY="$ENV_DIR/bin/python"
# sphn's bundled Opus declares an old CMake minimum; NGC CMake 4 rejects it.
# Keep this build tool inside the venv and ahead of the container binary.
"$PY" -m pip install --ignore-installed --no-deps 'cmake==3.31.6'
export PATH="$ENV_DIR/bin:$PATH" CARGO_BUILD_JOBS=8
# pip build isolation prepends its own CMake 4; Rust's cmake crate honors
# this explicit binary path even inside that temporary build environment.
export CMAKE
CMAKE=$("$PY" -c 'import cmake; print(cmake.CMAKE_BIN_DIR + "/cmake")')
"$CMAKE" --version
"$PY" -m evals.moshi_environment_setup \
    --requirements "$DL/repo/evals/moshi-rag-requirements.txt" --out "$OUT"
# Build against this exact NGC Torch ABI; never install a replacement Torch.
export MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=9.0 XFORMERS_DISABLE_FLASH_ATTN=1
timeout 1500 "$PY" -m pip install --no-deps --no-build-isolation "$ATTENTION"
"$PY" -m pip install --no-deps --no-build-isolation "$SOURCE/moshi"
"$PY" -m evals.moshi_environment_setup --verify \
    --requirements "$DL/repo/evals/moshi-rag-requirements.txt" --out "$OUT"
printf '%s\n' "$ENV_DIR" > "$OUT/env-path.txt"
CONTAINER
