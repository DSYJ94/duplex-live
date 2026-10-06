#!/bin/bash
# THE MIC DEMO (2026-10-06) — talk to the model live.
#
#   mic -> ssh -> cluster live loop (80 ms ticks) -> speaker
#
# You talk; the model decides WHEN to take its turns (that is what we
# trained and verified); the WORDS it speaks are the scripted answers
# of the demo lesson, in order (word content is out of scope for this
# phase — the honest claim is turn-taking, see EXPERIMENTS 2026-10-05).
# Speak in turns with pauses; interrupt it any time (it yields).
#
# Needs: sox on this Mac (brew install sox). End with Ctrl-C.

set -euo pipefail
command -v sox >/dev/null || { echo "sox missing: brew install sox"; exit 1; }

LESSON=${LESSON:-dense_dialog-116}   # the answer bank (verified lesson)

sox -d -t raw -r 16000 -b 16 -c 1 -e signed-integer - \
| ssh u6si.aip2.isambard \
  "cd /projects/u6si/duplex-live/repo && \
   PYTHONPATH=/projects/u6si/duplex-live/pylibs:\$PWD \
   srun -p workq -A brics.u6si --gres=gpu:1 --time=00:30:00 \
   --job-name=dl-micdemo --export=NONE \
   apptainer exec --nv --bind /projects,/lus,/home \
     --env PYTHONPATH=/projects/u6si/duplex-live/pylibs:/projects/u6si/duplex-live/repo \
     --env HF_HOME=/projects/u6si/duplex-live/models/hf-cache \
     --env TMPDIR=/projects/u6si/duplex-live/tmp \
     /projects/u6si/containers/latentsprites-pytorch-25.05.sif \
   python -u -m m2.live \
     --ckpt /projects/u6si/duplex-live/runs/scale1-q2-pt1/p1_checkpoint.pt \
     --batch /projects/u6si/duplex-live/runs/synth-scale1-q/audio \
     --scenarios $LESSON \
     --gate-ckpt /projects/u6si/duplex-live/runs/scale1-q2-gate1/gate_checkpoint.pt \
     --gate-threshold 0.10 --gate-grace 6 \
     --source stdin --sink stdout \
     --out /projects/u6si/duplex-live/runs/live-mic --realtime" \
| sox -t raw -r 16000 -b 16 -c 1 -e signed-integer - -d
