#!/bin/bash
#SBATCH --job-name=dl-fresh-concurrent-hybrid
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/dl-fresh-concurrent-hybrid-%j.out
set -euo pipefail
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable commit required}"
: "${KYUTAI_STT_ASSETS_JOB:?Pinned ASR assets required}"
: "${KYUTAI_TTS_ASSETS_JOB:?Pinned TTS assets required}"
: "${KYUTAI_TTS_MIMI_LOAD_JOB:?Strict codec proof required}"
for JOB in "$KYUTAI_STT_ASSETS_JOB" "$KYUTAI_TTS_ASSETS_JOB" "$KYUTAI_TTS_MIMI_LOAD_JOB"; do [[ "$JOB" =~ ^[0-9]+$ ]]; done
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
set -a
source "$DL/keys/openai.env"
set +a
 : "${DUPLEX_FRESH_INPUTS_JOB:?Fresh user inputs required}"
: "${DUPLEX_FRESH_QUALITY_JOB:?Independent input quality required}"
[[ "$DUPLEX_FRESH_INPUTS_JOB" =~ ^[0-9]+$ && "$DUPLEX_FRESH_QUALITY_JOB" =~ ^[0-9]+$ ]]
INPUTS="$DL/runs/duplex-fresh-inputs-$DUPLEX_FRESH_INPUTS_JOB"
QUALITY="$DL/runs/duplex-fresh-quality-$DUPLEX_FRESH_QUALITY_JOB/quality.json"
mapfile -t IDS < <(python3 - "$QUALITY" <<'PYMETA'
import json,sys
q=json.load(open(sys.argv[1]))
for row in q['results']:
 if row['passed']:print(row['scenario_id'])
PYMETA
)
if [ "${#IDS[@]}" -eq 0 ]; then echo 'No independently approved fresh input'; exit 1; fi
if [ -n "${DUPLEX_FRESH_SCENARIOS:-}" ]; then
 IFS=',' read -r -a IDS <<< "$DUPLEX_FRESH_SCENARIOS"
 for ID in "${IDS[@]}"; do [[ "$ID" =~ ^hybrid-fresh-[0-9]{3}$ ]]; done
fi
EXTRA=()
if [ "${DUPLEX_TOOL_CONTEXT:-0}" = 1 ]; then EXTRA+=(--include-tool-context); fi
if [ "${DUPLEX_SCOPE_READ_TOOLS:-0}" = 1 ]; then EXTRA+=(--scope-read-tools); fi
if [ -n "${DUPLEX_FRESH_VALIDATOR_SOURCE:-}" ]; then
 : "${DUPLEX_FRESH_VALIDATOR_COMMIT:?Historical validator commit required}"
 EXTRA+=(--input-validator-source "$DUPLEX_FRESH_VALIDATOR_SOURCE" --input-validator-commit "$DUPLEX_FRESH_VALIDATOR_COMMIT")
fi
SCENARIOS=()
for ID in "${IDS[@]}"; do SCENARIOS+=(--scenario "$ID"); done
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 \
 "$DL/pytorch-new.sif" python -u -m evals.duplex_hybrid_probe \
 --stt-assets "$DL/runs/kyutai-stt-assets-$KYUTAI_STT_ASSETS_JOB/assets.json" \
 --tts-assets "$DL/runs/kyutai-tts-assets-$KYUTAI_TTS_ASSETS_JOB/assets.json" \
 --mimi-load-approval "$DL/runs/kyutai-tts-mimi-load-$KYUTAI_TTS_MIMI_LOAD_JOB/report.json" \
 --data "$INPUTS" --input-profile fresh-evaluation --input-quality "$QUALITY" "${SCENARIOS[@]}" \
 --controller-protocol "${DUPLEX_CONTROLLER_PROTOCOL:-required-action}" --evaluation-use "${DUPLEX_EVALUATION_USE:-fresh}" "${EXTRA[@]}" --interrupt-policy "${DUPLEX_INTERRUPT_POLICY:-acoustic}" --lookup-delay-s 12 --out "$DL/runs/duplex-hybrid-probe-${SLURM_JOB_ID}"
