#!/bin/bash
#SBATCH --job-name=dl-native-fresh2-evaluate
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:25:00
#SBATCH --output=logs/dl-native-fresh2-evaluate-%A_%a.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${NATIVE_FRESH_INPUTS_JOB:?Full20 user inputs required}"
: "${NATIVE_FRESH_QUALITY_JOB:?Full20 independent source quality required}"
: "${NATIVE_FRESH_CANDIDATE_JOB:?Fixed independently restored candidate required}"
: "${NATIVE_FRESH_BEHAVIOR_REVIEW:?Exact committed review of all10 development cases required}"
: "${SLURM_ARRAY_JOB_ID:?Complete 0-39 array, no more than two concurrent tasks}"
for NAME in NATIVE_FRESH_INPUTS_JOB NATIVE_FRESH_QUALITY_JOB NATIVE_FRESH_CANDIDATE_JOB SLURM_ARRAY_TASK_ID; do
 [[ "${!NAME}" =~ ^[0-9]+$ ]]
done
[[ "$SLURM_ARRAY_TASK_ID" -lt 40 ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
readarray -t FIELDS < <(python3 - "$NATIVE_FRESH_CANDIDATE_JOB" "$NATIVE_FRESH_BEHAVIOR_REVIEW" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json,sys
from pathlib import Path
b=Path('/projects/u6si/duplex-live');candidate=sys.argv[1];proof=b/'runs'/('context-training-proof-'+candidate)
r=json.loads((proof/'report.json').read_text());reload=json.loads((proof/'reload-report.json').read_text())
assert r['passed'] and r['optimizer_steps'] in (8,40) and r['additional_native_v2']
if r['optimizer_steps']==40:
 assert r['additional_native_v2']['complete24_per8'] and r['initial_optimizer_steps']==8
 assert r['updates_this_run']==32 and r['resumed_from_training_state']
assert reload['passed'] and reload['fresh_process'] and reload['optimizer_rng_state_restored']
# A reviewer must assess all10 native tasks; neither loss nor source/input audits can supply this record.
p=Path(sys.argv[2]).resolve();assert Path.cwd().resolve() in p.parents
review=json.loads(p.read_text())
assert review['schema']=='dl-native-v2-development-review-v1' and review['candidate_job']==candidate
assert review['all10_native_cases_reviewed'] and len(review['cases'])==10
assert review['fresh_evaluation_authorized'] and not review['checkpoint_adopted']
plan=json.loads(Path('evals/fixtures/native-fulltask-qualification-v2-20261019.json').read_text());i=int(sys.argv[3]);case=plan['cases'][i%20]
kind={'ordinary':'simple','quiet':'simple','pause':'simple'}.get(case['kind'],case['kind'])
print('\n'.join([case['id'],kind,case['backend'],str(case.get('minimum_actual_lookup_delay_s',0.)),
                '7510331' if i<20 else candidate]))
PY
)
test "${#FIELDS[@]}" = 5
EXTRA=()
if [ "${FIELDS[2]}" = native-asr ]; then
 EXTRA+=(--asr-python "$DL/envs/moshi-annotation-7379026/bin/python")
 set -a
 source "$DL/keys/openai.env"
 set +a
else
 export OPENAI_API_KEY=
fi
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
 "$DL/pytorch-new.sif" python -u -m evals.context_spanning_fresh_probe \
 --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --data "$DL/runs/native-fresh2-inputs-$NATIVE_FRESH_INPUTS_JOB" \
 --native-fresh-candidate-proof "$DL/runs/context-training-proof-$NATIVE_FRESH_CANDIDATE_JOB" \
 --native-fresh-behavior-review "$NATIVE_FRESH_BEHAVIOR_REVIEW" \
 --native-fresh-quality "$DL/runs/native-fresh2-judge-$NATIVE_FRESH_QUALITY_JOB/quality.json" \
 --scenario "${FIELDS[0]}" --expected-kind "${FIELDS[1]}" --backend-mode "${FIELDS[2]}" \
 --lookup-delivery-delay-s "${FIELDS[3]}" --adapter-proof "$DL/runs/context-training-proof-${FIELDS[4]}" \
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
 --out "$DL/runs/native-fresh2-evaluate-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID" "${EXTRA[@]}"
# Complete40 actual comparisons and independent speech/behavior/causality review are still required.
