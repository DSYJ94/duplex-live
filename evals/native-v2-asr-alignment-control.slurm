#!/bin/bash
#SBATCH --job-name=dl-native-v2-asr-alignment
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dl-native-v2-asr-alignment-%j.out
set -euo pipefail
ulimit -c 0
DL=/projects/u6si/duplex-live
: "${DL_TASK_SOURCE:?Immutable source required}"
: "${DL_TASK_COMMIT:?Immutable identity required}"
test "$(git -C "$DL_TASK_SOURCE" rev-parse HEAD)" = "$DL_TASK_COMMIT"
test -z "$(git -C "$DL_TASK_SOURCE" status --porcelain)"
cd "$DL_TASK_SOURCE"
run() {
 apptainer exec --nv --bind /projects,/lus,/home \
  --env PYTHONPATH="$DL_TASK_SOURCE" --env PYTHONDONTWRITEBYTECODE=1 --env OMP_NUM_THREADS=8 \
  --env HF_HOME="$DL/models/hf-cache" --env OPENAI_API_KEY= \
  "$DL/pytorch-new.sif" "$DL/envs/moshi-annotation-7379026/bin/python" -u "$@"
}
OUT="$DL/runs/native-v2-asr-alignment-$SLURM_JOB_ID"
run -m evals.moshi_speech_audit --out "$OUT" --trim-leading-silence --trim-trailing-silence \
 --live-runs "$DL/runs/native-v2-teacher-7541876/native2-train-008" \
 "$DL/runs/native-v2-teacher-7541876/native2-train-015"
run -c '
import json,sys
from pathlib import Path
old={Path(x["run"]).name:x for x in json.load(open("/projects/u6si/duplex-live/runs/native-v2-teacher-audit-7541878/report.json"))["results"]}
rows=json.load(open(sys.argv[1]))["results"]
assert len(rows)==2
for x in rows:
 prior=old[Path(x["run"]).name]
 assert x["wav_sha256"]==prior["wav_sha256"] and x["model_text"]==prior["model_text"]
 assert x["timestamps_restored_to_original_wav"] and not x["internal_silence_removed"]
 assert x["asr_time_bounds_valid"] and x["asr_word_bounds_valid"] and x["word_starts_within_recorded_audio"]
 assert min(w["start"] for s in prior["asr_segments"] for w in s["words"])<x["first_nonzero_wav_sample_s"]-.2
 print(json.dumps(dict(run=x["run"],same_wav=True,old_onset_invalid=True,new_onset_inside_recorded_audio=True,wer=x["wer_against_model_text"])))
' "$OUT/report.json"
# This tests measurement only. It grants neither teacher-corpus nor model approval.
