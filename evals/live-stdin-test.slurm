#!/bin/bash
#SBATCH --job-name=dl-livestdin
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/dl-livestdin-%j.out

# STDIN/STDOUT PIPE TEST (2026-10-06): exactly the demo pipe, with the
# user stem standing in for the mic: cat raw | m2.live --source stdin
# --sink stdout > raw. Proves the streaming legs (int16 chunking, logs-
# to-stderr, audio-out purity) before the owner's mic is attached.

set -euo pipefail

DL=/projects/u6si/duplex-live
REPO=$DL/repo
export HF_HOME=$DL/models/hf-cache TMPDIR=$DL/tmp PYTHONPATH=$DL/pylibs:$REPO
OUT=$DL/runs/live-stdin-test
SID=dense_dialog-116

cd "$REPO"
mkdir -p "$OUT"

# stem -> raw int16 16k mono (the mic format)
apptainer exec --bind /projects,/lus,/home --env PYTHONPATH=$PYTHONPATH \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python - <<PYEOF
import numpy as np, soundfile as sf, sys
sys.path.insert(0, ".")
x, sr = sf.read("$DL/runs/synth-scale1-q/audio/$SID/stem_user.wav",
               dtype="float32")
if x.ndim > 1: x = x.mean(axis=1)
if sr != 16000:
    import librosa; x = librosa.resample(x, orig_sr=sr, target_sr=16000)
(x * 32767).astype("<i2").tofile("$OUT/mic.raw")
print("mic.raw", len(x) / 16000, "s")
PYEOF

cat "$OUT/mic.raw" | apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$PYTHONPATH --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python -u -m m2.live \
    --ckpt $DL/runs/scale1-q2-pt1/p1_checkpoint.pt \
    --batch $DL/runs/synth-scale1-q/audio \
    --scenarios $SID \
    --gate-ckpt $DL/runs/scale1-q2-gate1/gate_checkpoint.pt \
    --gate-threshold 0.10 --gate-grace 6 \
    --source stdin --sink stdout --out "$OUT" --realtime \
    > "$OUT/model.raw"

# raw -> wav + a check that stdout carried ONLY audio (logs on stderr)
apptainer exec --bind /projects,/lus,/home --env PYTHONPATH=$PYTHONPATH \
    /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
    python - <<PYEOF
import numpy as np, soundfile as sf
x = np.fromfile("$OUT/model.raw", dtype="<i2")
print("model.raw:", len(x) / 16000.0, "s, expected == mic duration")
sf.write("$OUT/${SID}_model.wav", x.astype("float32") / 32768.0, 16000)
peak = float(np.abs(x).max()) if len(x) else 0.0
print("peak:", peak, "(0.0 = silent stream = bug)")
PYEOF
echo "stdin/stdout pipe test done: $OUT"
