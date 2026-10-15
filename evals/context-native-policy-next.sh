#!/bin/bash
# Run manually from an authorised Isambard login after reviewing this file.
# Submit one bounded pilot, one TRAIN diagnostic and eight dependent replays.
set -euo pipefail
DL=/projects/u6si/duplex-live
PIN=9b411d0f90b3d83608bfbbece4bdd912871b5d59
SOURCE="$DL/worktrees/recovery-$PIN"
PROOF="$DL/runs/context-training-proof-7457079"
RECEIPTS="$DL/runs/context-native-policy-next-7457079"
test "$(git -C "$SOURCE" rev-parse HEAD)" = "$PIN"
test -z "$(git -C "$SOURCE" status --porcelain)"
python3 - "$PROOF" "$SOURCE" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
proof, source = map(Path, sys.argv[1:])
report = json.loads((proof / 'report.json').read_text())
reload = json.loads((proof / 'reload-report.json').read_text())
history = json.loads((proof / 'history-audit.json').read_text())
assert report['passed'] and report['optimizer_steps'] == 8
assert report['frozen_weights_unchanged'] and report['zero_adapter_all_valid_logits_exact']
assert report['native_depth_parity']['passed'] and report['canonical_logits_dtype'] == 'float32'
assert report['text_padding_convention'] == 'context-pad3'
assert report['request_policy'] == 'native-prelead'
assert reload['passed'] and reload['fresh_process'] and reload['all_valid_logits_exact']
assert history['candidate_passed'] and history['results']['aligned_candidate']['temporal_parity']['passed']
for file, key in [('adapters.safetensors', 'checkpoint_sha256'), ('data.json', 'data_sha256'),
                  ('voice.pt', 'voice_sha256')]:
    assert hashlib.sha256((proof / file).read_bytes()).hexdigest() == report[key]
assert report['checkpoint_sha256'] == reload['checkpoint_sha256'] == (
    '0b20cc01db2174fb114201f81e41cdf0ddeb5373d3f72edfc3efca898ba59716')
assert report['data_sha256'] == history['data_sha256'] == (
    '7c0a3c6f42675766b0a3edb4569dcad1ad40a8619a21215d11330107d9926aef')
assert hashlib.sha256((source / 'evals/context_native_layout.py').read_bytes()).hexdigest() == (
    history['layout_code_sha256'])
for file, expected in {
    'context_training_proof.py': 'e4e8a90638f125c9ea81ba58d685ba90',
    'context_training_data.py': '795ccb60eff56a3da16012c2c42ba7dc',
    'context_request_policy.py': '86963c04747518287306f04e55b20554',
    'context_spanning_probe.py': '52f6b4db8f6c1765991da97ba45993bf',
}.items():
    assert hashlib.md5((source / 'evals' / file).read_bytes()).hexdigest() == expected
print('Verified native-policy preflight and immutable source.')
PY
# A repeated invocation stops here, preserving any partial submission receipt.
mkdir "$RECEIPTS"
cd "$DL/repo"
COMMON="ALL,DL_TASK_SOURCE=$SOURCE,DL_TASK_COMMIT=$PIN,CONTEXT_TEXT_PADDING=context-pad3,CONTEXT_REQUEST_POLICY=native-prelead"
PILOT=$(sbatch --parsable --time=00:15:00 \
 --export="$COMMON,CONTEXT_PILOT=1,CONTEXT_PREFLIGHT_JOB=7457079,CONTEXT_LAYOUT_AUDIT_JOB=7457079" \
 "$SOURCE/evals/context-training-proof.slurm")
[[ "$PILOT" =~ ^[0-9]+$ ]]
printf '%s\n' "$PILOT" > "$RECEIPTS/pilot-job.txt"
sbatch --parsable --time=00:10:00 \
 --export="$COMMON,CONTEXT_TEACHER_AUDIT=1,CONTEXT_TEACHER_AUDIT_JOBS=7457079" \
 "$SOURCE/evals/context-training-proof.slurm" > "$RECEIPTS/calibration-job.txt"
NATIVE="ALL,DL_TASK_SOURCE=$SOURCE,DL_TASK_COMMIT=$PIN,CONTEXT_ASSETS_JOB=7454205,CONTEXT_ADAPTER_JOB=$PILOT,CONTEXT_LAYOUT_AUDIT_JOB=7457079,CONTEXT_VOICE=accepted-synthetic,CONTEXT_BACKEND_MODE=native-asr"
for SCENE in 000 001; do
 for VALUE in 'yellow circle' 'purple diamond'; do
  JOB=$(sbatch --parsable --gres=gpu:2 --time=00:10:00 --dependency="afterok:$PILOT" \
   --export="$NATIVE,CONTEXT_SCENARIO=recovery-validation-$SCENE,CONTEXT_PRIVATE_VALUE=$VALUE" \
   "$SOURCE/evals/context-spanning-probe.slurm")
  printf '%s\t%s\t%s\n' "$JOB" "$SCENE" "$VALUE" >> "$RECEIPTS/native-jobs.tsv"
 done
done
for SCENE in 004 005 006 007; do
 JOB=$(sbatch --parsable --gres=gpu:2 --time=00:10:00 --dependency="afterok:$PILOT" \
  --export="$NATIVE,CONTEXT_SCENARIO=recovery-validation-$SCENE,CONTEXT_PRIVATE_VALUE=yellow circle" \
  "$SOURCE/evals/context-spanning-probe.slurm")
 printf '%s\t%s\t%s\n' "$JOB" "$SCENE" 'yellow circle' >> "$RECEIPTS/native-jobs.tsv"
done
cat "$RECEIPTS/pilot-job.txt" "$RECEIPTS/calibration-job.txt" "$RECEIPTS/native-jobs.tsv"
