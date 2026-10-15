#!/bin/bash
#SBATCH --job-name=dl-persona-train
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=110G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-persona-train-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
cd "$DL/repo"
test "$(git -C "$DL/personaplex" rev-parse HEAD)" = 3428dfd95309a7f3c84fd93259ded0f810d1ff91
test -z "$(git -C "$DL/personaplex" status --porcelain)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ARGS=(--cache "$DL/runs/recovery-mixed-codes-7411513")
if [ -n "${PERSONA_EXTRA_CACHE:-}" ]; then ARGS+=(--cache "$PERSONA_EXTRA_CACHE"); fi
if [ -n "${PERSONA_CONTROL_CACHE:-}" ]; then ARGS+=(--cache "$PERSONA_CONTROL_CACHE"); fi
if [ "${PERSONA_FREEZE_LANGUAGE:-0}" = 1 ]; then ARGS+=(--freeze-language); fi
if [ "${PERSONA_PAIRED_GROUNDING:-0}" = 1 ]; then ARGS+=(--paired-grounding); fi
if [ "${PERSONA_FEEDBACK:-0}" = 1 ]; then ARGS+=(--control-feedback --freeze-policy); fi
if [ -n "${PERSONA_INIT_CHECKPOINT:-}" ]; then ARGS+=(--init-checkpoint "$PERSONA_INIT_CHECKPOINT"); fi
ARGS+=(--model "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309"
 --voice "$DL/runs/voice-quality-7414595/codec/reference.wav"
 --template "$DL/runs/personaplex-bridge-7428894"
 --proof "$DL/runs/personaplex-history-7434702"
 --steps "${PERSONA_STEPS:-8}" --out "${PERSONA_RELOAD_DIR:-$DL/runs/personaplex-train-${SLURM_JOB_ID}}")
MODES=(train reload)
if [ "${PERSONA_RELOAD_ONLY:-0}" = 1 ]; then
 : "${PERSONA_RELOAD_DIR:?Existing checkpoint directory required}"
 MODES=(reload)
elif [ -n "${PERSONA_RELOAD_DIR:-}" ]; then
 echo 'Existing checkpoint directory can only be used for reload' >&2
 exit 1
fi
for MODE in "${MODES[@]}"; do
 EXTRA=()
 if [ "$MODE" = reload ]; then EXTRA=(--reload); fi
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/personaplex/moshi:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL/repo" \
  --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  "$DL/pytorch-new.sif" python -u -m evals.personaplex_train "${ARGS[@]}" "${EXTRA[@]}"
done
