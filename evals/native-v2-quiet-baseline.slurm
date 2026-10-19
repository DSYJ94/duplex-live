#!/bin/bash
#SBATCH --job-name=dl-native-v2-quiet-baseline
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=00:25:00
#SBATCH --output=logs/dl-native-v2-quiet-baseline-%A_%a.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable source required}"
: "${SLURM_ARRAY_JOB_ID:?Complete0-1 baseline array required}"
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[01]$ ]]
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/projects/u6si/duplex-live/runs/context-training-proof-7510331')
r=json.loads((p/'report.json').read_text());loaded=json.loads((p/'reload-report.json').read_text())
assert r['passed'] and r['optimizer_steps']==520 and len(r['retention_teacher_text']['teachers'])==25
assert loaded['passed'] and loaded['fresh_process'] and loaded['optimizer_rng_state_restored']
assert len(json.loads((p/'data.json').read_text())['components'])==28
PY
export OPENAI_API_KEY=
SCENE=native2-train-014
if [ "$SLURM_ARRAY_TASK_ID" = 1 ]; then SCENE=native2-train-015; fi
OUT="$DL/runs/native-v2-quiet-baseline-${SLURM_ARRAY_JOB_ID}_$SLURM_ARRAY_TASK_ID"
apptainer exec --nv --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL/tmp/voice-quality-7415052:$DL/envs/moshi-rag-7379661/lib/python3.12/site-packages:$DL_TASK_SOURCE" \
 --env HF_HOME="$DL/models/hf-cache" --env HF_HUB_OFFLINE=1 --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 "$DL/pytorch-new.sif" python -u -m evals.context_spanning_probe \
 --assets "$DL/runs/context-spanning-assets-7454205/manifest.json" \
 --base "$DL/models/hf-cache/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309" \
 --data "$DL/runs/native-curriculum-v2-data-7541875" --scenario "$SCENE" --expected-kind simple \
 --private-value 'red circle' --backend-mode observe-only --train-history \
 --adapter-proof "$DL/runs/context-training-proof-7510331" --voice accepted-synthetic \
 --voice-reference "$DL/runs/voice-quality-7414595/codec/reference.wav" \
 --voice-provenance "$DL/runs/voice-quality-7414595/codec/report.json" --out "$OUT"
python3 - "$OUT" <<'PYCODE'
import json,sys,hashlib
from pathlib import Path
p=Path(sys.argv[1]);plan=Path('evals/fixtures/native-v2-development-20261019.json')
(p/'evaluation-only.json').write_text(json.dumps(dict(schema='dl-native-v2-quiet-baseline-v1',
    do_not_train=True,training_approved=False,baseline_job='7510331',
    development_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    fresh_qualification=False),indent=2)+'\n')
PYCODE
apptainer exec --bind /projects,/lus,/home \
 --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 --env OPENAI_API_KEY= \
 --env HF_HOME="$DL/models/hf-cache" \
 "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u -m evals.moshi_speech_audit \
 --runs "$OUT" --device cpu --trim-leading-silence --trim-trailing-silence --keep-silent-failures --out "$OUT/speech"
# Reused TRAIN/development baseline only. Raw history is unapproved/masked, never new labels.
