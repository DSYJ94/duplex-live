#!/bin/bash
#SBATCH --job-name=dl-native-v2-spelling-reaudit
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dl-native-v2-spelling-reaudit-%j.out
set -euo pipefail
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${NATIVE_TEACHER_JOB:?Original unchanged complete24 clarity corpus required}"
test "$NATIVE_TEACHER_JOB" = 7544225
export NATIVE_ASR_DEVICE=cpu NATIVE_ORTHOGRAPHIC_AGREEMENT=1
source "$DL_TASK_SOURCE/evals/native-v2-teacher-audit.slurm"
python3 - "$DL/runs/native-v2-teacher-audit-$SLURM_JOB_ID" <<'PY'
import json,sys,hashlib
from pathlib import Path
out=Path(sys.argv[1]);prior=Path('/projects/u6si/duplex-live/runs/native-v2-teacher-audit-7544226')
old={Path(r['run']).name:r for r in json.loads((prior/'report.json').read_text())['results']}
new={Path(r['run']).name:r for r in json.loads((out/'report.json').read_text())['results']}
before={r['scenario_id']:r for r in json.loads((prior/'teacher_checks.json').read_text())['results']}
after={r['scenario_id']:r for r in json.loads((out/'teacher_checks.json').read_text())['results']}
assert len(old)==24 and old.keys()==new.keys()==before.keys()==after.keys()
changed=[]
for sid in old:
 row=dict(new[sid]);policy=row.pop('word_agreement')
 assert row==old[sid], 'Raw WAV/model/ASR/word timestamps or legacy diagnostic changed: '+sid
 assert after[sid]['speech_and_fact_checks_pass']
 for key in ('measured_words','reference_arrivals','delegation_events','raw_record'):
  assert before[sid][key]==after[sid][key], (sid,key)
 if before[sid]['speech_and_fact_checks_pass']!=after[sid]['speech_and_fact_checks_pass']:changed.append(sid)
assert changed==['native2-train-022']
last=new[changed[0]]
assert last['wer_against_model_text']>.08 and last['word_agreement']['canonical_word_wer']==0.
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=dict(schema='dl-same-wav-finite-spelling-control-v1',passed=True,cases=24,
 prior_report_sha256=digest(prior/'report.json'),new_report_sha256=digest(out/'report.json'),
 raw_wav_model_asr_words_times_receipts_unchanged=True,prior23_passes_unchanged=True,
 changed_verdicts=changed,raw_wer_retained=last['wer_against_model_text'],
 canonical_word_wer=last['word_agreement']['canonical_word_wer'],
 same_wer_threshold=.08,same_exact_spoken_words_required=True,training_approved=False)
(out/'same-wav-spelling-control.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
PY
