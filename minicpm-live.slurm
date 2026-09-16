#!/bin/bash
#SBATCH --job-name=minicpm-live
#SBATCH --partition=workq
#SBATCH --account=brics.u6si
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/minicpm-live-%j.out
#
# MiniCPM-o 4.5 full-duplex live demo (bare-metal port of the docker-compose
# three-stage flow into the Apptainer container):
#   Browser -> gateway.py (:8006 https) -> worker.py (:22400) -> py_backend.server (:22500)
#
# Client tunnel:  ssh -L 8006:<node>:8006 u6si.aip2.isambard
# Browser:        https://localhost:8006  (accept self-signed cert)

set -euo pipefail

DL=/projects/u6si/duplex-live
export HF_HOME=$DL/models/hf-cache
export TMPDIR=$DL/tmp
# self-contained overlay for the demo (needs numpy>=2.2; pylibs-minicpm pins 1.26.4)
OVERLAY=$DL/pylibs-mcpmdemo
CONTAINER=/projects/u6si/containers/latentsprites-pytorch-25.05.sif
DEMO=$DL/mcpm-demo

cd "$DEMO"
[ -f config.json ] || cp config.example.json config.json
mkdir -p certs data
if [ ! -f certs/cert.pem ]; then
    openssl req -x509 -newkey rsa:2048 -nodes -days 365 \
        -keyout certs/key.pem -out certs/cert.pem -subj "/CN=minicpm-o"
fi

APPT="apptainer exec --nv --bind /projects,/lus,/home \
    --env PYTHONPATH=$OVERLAY \
    --env HF_HOME=$HF_HOME --env TMPDIR=$TMPDIR \
    --env HF_TOKEN=$(cat $DL/.hf_token) \
    --env PIP_CONSTRAINT=/dev/null \
    $CONTAINER"

# ---- deps (once) ----
if [ ! -f "$OVERLAY/.minicpm-demo-deps-ok" ]; then
    echo "[setup] installing demo requirements into $OVERLAY"
    # NGC container pins numpy==1.26.4 via /etc/pip/constraint.txt; demo needs numpy>=2.2
    $APPT pip install --target="$OVERLAY" -r requirements.txt
    # never let pip shadow the container torch stack
    rm -rf "$OVERLAY"/torch "$OVERLAY"/torch-* "$OVERLAY"/nvidia_* "$OVERLAY"/triton*
    touch "$OVERLAY/.minicpm-demo-deps-ok"
fi

NODE=$(hostname)
echo "[live] node=$NODE  gateway https://$NODE:8006"

# ---- 1. backend (loads model, ~90s) ----
$APPT python -m py_backend.server \
    --host 0.0.0.0 --port 22500 \
    --gpu-id 0 \
    --model-path openbmb/MiniCPM-o-4_5 &
backend_pid=$!

echo "[live] waiting for backend /health ..."
for i in $(seq 1 300); do
    if ! kill -0 "$backend_pid" 2>/dev/null; then
        echo "[live] ERROR: backend exited during model load" >&2
        exit 1
    fi
    if curl -sf http://127.0.0.1:22500/health >/dev/null 2>&1; then
        echo "[live] backend ready (~$((i*2))s)"
        break
    fi
    sleep 2
    [ "$i" -eq 300 ] && { echo "[live] ERROR: backend not ready in 600s" >&2; exit 1; }
done

# ---- 2. worker (pure forwarding) ----
$APPT python worker.py \
    --host 0.0.0.0 --port 22400 \
    --gpu-id 0 \
    --backend-server-url http://127.0.0.1:22500 &
worker_pid=$!
sleep 3
curl -sf http://127.0.0.1:22400/health >/dev/null 2>&1 \
    && echo "[live] worker ready" || echo "[live] WARN: worker /health not ready yet"

# ---- 3. gateway (https entrypoint) ----
$APPT python gateway.py \
    --host 0.0.0.0 --port 8006 --internal-port 8007 \
    --https --ssl-certfile certs/cert.pem --ssl-keyfile certs/key.pem &
gateway_pid=$!
sleep 3

# register worker with gateway
for i in $(seq 1 30); do
    if curl -sf -X PUT -H "content-type: application/json" \
        --data "{\"endpoint\":\"$NODE:22400\",\"gpu_group\":\"gpu-0\"}" \
        http://127.0.0.1:8007/internal/workers/minicpm-0 >/dev/null 2>&1; then
        echo "[live] worker registered with gateway"
        break
    fi
    sleep 2
done

echo "[live] all stages up. backend=$backend_pid worker=$worker_pid gateway=$gateway_pid"
echo "[live] tunnel: ssh -L 8006:$NODE:8006 u6si.aip2.isambard  ->  https://localhost:8006"

cleanup() {
    kill "$gateway_pid" "$worker_pid" "$backend_pid" 2>/dev/null || true
    wait 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

while true; do
    for p in "$backend_pid" "$worker_pid" "$gateway_pid"; do
        kill -0 "$p" 2>/dev/null || { echo "[live] process $p died, shutting down" >&2; cleanup; }
    done
    sleep 5
done
