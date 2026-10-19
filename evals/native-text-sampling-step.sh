#!/bin/bash
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
INDEX="${1:?One of eight paired diagnostic cases required}"
[[ "$INDEX" =~ ^[0-7]$ && "${NATIVE_SAMPLING_CANDIDATE_JOB:?}" =~ ^[0-9]+$ ]]
: "${DL_TASK_SOURCE:?Immutable source required}" "${DL_TASK_COMMIT:?}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
CASE="$DL/runs/native-text-sampling-case-${SLURM_JOB_ID}_$INDEX.json"
python3 - "$INDEX" "$CASE" <<'PY'
import os,json,sys
from evals.native_text_sampling_diagnostic import case
r=case(int(sys.argv[1]),os.environ['NATIVE_SAMPLING_CANDIDATE_JOB'],float(os.environ.get('NATIVE_SAMPLING_TEXT_TEMPERATURE','0')))
r.update(allocation_job=os.environ['SLURM_JOB_ID'],step=os.environ['SLURM_STEP_ID'],
    assigned_cuda_devices=os.environ['CUDA_VISIBLE_DEVICES'],source=os.environ['DL_TASK_COMMIT'])
assert len(r['assigned_cuda_devices'].split(','))==1 and r['step'].isdigit()
with open(sys.argv[2],'x') as f:json.dump(r,f,indent=2)
PY
readarray -t FIELDS < <(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["scenario"]); print(r["checkpoint_job"])' "$CASE")
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -m evals.context_spanning_probe \
 --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --data "$DL/runs/recovery-data-7407554" --scenario "${FIELDS[0]}" --expected-kind simple \
 --backend-mode observe-only --train-history --native-sampling-diagnostic --native-text-temperature "${NATIVE_SAMPLING_TEXT_TEMPERATURE:-0}" \
 --private-value 'red circle' --adapter-proof "$DL/runs/context-training-proof-${FIELDS[1]}" \
 --voice accepted-synthetic --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" \
 --out "$DL/runs/native-text-sampling-${SLURM_JOB_ID}_$INDEX"
# Captured histories are diagnostic-only; no optimizer or data approval is implied.
