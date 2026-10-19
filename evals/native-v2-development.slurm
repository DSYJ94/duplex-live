#!/bin/bash
#SBATCH --job-name=dl-native-v2-development
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:25:00
#SBATCH --output=logs/dl-native-v2-development-%A_%a.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable committed source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
: "${NATIVE_V2_CANDIDATE_JOB:?Independently restored eight-update candidate required}"
: "${SLURM_ARRAY_JOB_ID:?Complete 0-9 array, at most two concurrent jobs, required}"
[[ "$NATIVE_V2_CANDIDATE_JOB" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ && "$SLURM_ARRAY_TASK_ID" -lt 10 ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
CASE="$DL/runs/native-v2-development-case-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID.json"
python3 - "$DL_TASK_SOURCE" "$NATIVE_V2_CANDIDATE_JOB" "$SLURM_ARRAY_TASK_ID" "$CASE" <<'PY'
import json,sys
from pathlib import Path
root=Path('/projects/u6si/duplex-live')
proof=root/'runs'/('context-training-proof-'+sys.argv[2])
r=json.loads((proof/'report.json').read_text());reload=json.loads((proof/'reload-report.json').read_text())
assert r['passed'] and r['optimizer_steps'] in (8,40) and r['additional_native_v2']
if r['optimizer_steps']==40:
 assert r['additional_native_v2']['complete24_per8'] and r['initial_optimizer_steps']==8
 assert r['updates_this_run']==32 and r['resumed_from_training_state']
assert reload['passed'] and reload['fresh_process'] and reload['optimizer_rng_state_restored']
spec=json.loads((Path(sys.argv[1])/'evals/fixtures/native-v2-development-20261019.json').read_text())
assert len(spec['cases'])==10 and spec['word_or_audio_forcing'] is False
Path(sys.argv[4]).write_text(json.dumps(spec['cases'][int(sys.argv[3])],indent=2)+'\n')
PY
readarray -t FIELDS < <(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("\n".join([r["scenario"],r["kind"],r["source"],r["backend"],r.get("value","red circle")]))' "$CASE")
DATA="$DL/runs/recovery-data-7407554"
if [ "${FIELDS[2]}" = native-v2 ]; then DATA="$DL/runs/native-curriculum-v2-data-7541875"; fi
EXTRA=()
if [ "${FIELDS[3]}" = observe-only ]; then
 EXTRA+=(--backend-mode observe-only --train-history)
else
 EXTRA+=(--backend-mode native-asr --asr-python "$DL/envs/moshi-annotation-7379026/bin/python")
fi
set -a
source "$DL/keys/openai.env"
set +a
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.context_spanning_probe \
 --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --data "$DATA" --scenario "${FIELDS[0]}" --expected-kind "${FIELDS[1]}" --private-value "${FIELDS[4]}" \
 --adapter-proof "$DL/runs/context-training-proof-$NATIVE_V2_CANDIDATE_JOB" \
 --voice accepted-synthetic --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
 --out "$DL/runs/native-v2-development-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID" "${EXTRA[@]}"
# Keep every raw result; a completed replay is not a task/quality/qualification pass.
