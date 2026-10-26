#!/usr/bin/env bash
set -euo pipefail
# Runs inside the pinned existing NGC image, with ONLY the new runtime writable.
test -f /src/portable/venus/requirements.txt
test ! -e /runtime/setup-complete.json
python3 -m venv /runtime/venv
/runtime/venv/bin/python -m pip install --disable-pip-version-check --no-cache-dir -r /src/portable/venus/requirements.txt
/runtime/venv/bin/python -m pip freeze > /runtime/pip-freeze.txt
/runtime/venv/bin/python - <<'PY'
import json, torch, transformers, stepaudio2, soundfile, librosa
assert torch.__version__.startswith('2.4.0')
assert transformers.__version__ == '4.51.3'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_properties(0).total_memory >= 39_000_000_000
with open('/runtime/setup-complete.json','x') as f:
    json.dump(dict(torch=torch.__version__,transformers=transformers.__version__,
        gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
        optimizer_updates=0,model_loaded=False),f)
PY
