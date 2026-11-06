#!/bin/bash
set -euo pipefail
BASE=/projects/u6ada/duplex-live
OUT=$BASE/runtime-preparation-v2
VENV=$BASE/envs/venus-arm64-v1
export TMPDIR=$BASE/tmp/venus-runtime-v2
mkdir -p "$TMPDIR"
python -m venv --system-site-packages "$VENV"
PY=$VENV/bin/python
"$PY" - <<'PY'
import importlib.metadata as m,json,torch
from pathlib import Path
p=Path('/projects/u6ada/duplex-live/runtime-preparation-v2')
versions={d.metadata['Name']:d.version for d in m.distributions() if d.metadata['Name'].lower().startswith(('torch','triton','nvidia-'))}
(p/'compiled-constraints.txt').write_text(''.join(k+'==='+v+'\n' for k,v in sorted(versions.items())))
assert torch.cuda.is_available() and torch.cuda.device_count()==1
(p/'base-runtime.json').write_text(json.dumps(dict(compiled_versions=versions,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),python=__import__('sys').version),indent=2)+'\n')
PY
# The immutable container owns compiled Torch/CUDA. All package changes stay in
# this separate venv. The audio-only model does not use the optional video stack.
unset PIP_CONSTRAINT
export PIP_REQUIRE_VIRTUALENV=1
"$PY" -m pip install --no-cache-dir --disable-pip-version-check --no-input --no-deps 'minicpmo-utils==1.0.6'
"$PY" -m pip install --no-cache-dir --disable-pip-version-check --no-input -c "$OUT/compiled-constraints.txt" \
  'transformers==4.51.3' 'numpy==1.26.4' 'pillow==10.4.0' 'librosa==0.9.0' \
  'accelerate>=1.3,<2' 'sentencepiece==0.2.0' 'safetensors>=0.5,<1' \
  'resampy==0.4.3' 'setuptools>=77,<81' 'soundfile==0.12.1' \
  'onnxruntime==1.21.0' onnx 'HyperPyYAML==1.2.3' 'einops==0.8.1' \
  'fastapi>=0.115,<1' 'uvicorn>=0.30,<1' 'websockets>=13,<16'
"$PY" - <<'PY'
import importlib.metadata as m,json,hashlib,torch,torchaudio
from pathlib import Path
from transformers.dynamic_module_utils import get_class_from_dynamic_module
p=Path('/projects/u6ada/duplex-live/runtime-preparation-v2')
before=json.loads((p/'base-runtime.json').read_text())
assert torch.__version__==before['torch'] and torch.cuda.is_available() and torch.cuda.device_count()==1
assert m.version('transformers')=='4.51.3' and m.version('minicpmo-utils')=='1.0.6'
import stepaudio2
t=m.distribution('minicpmo-utils').locate_file('stepaudio2/token2wav.py')
assert hashlib.sha256(Path(t).read_bytes()).hexdigest()=='0598c25ad9883f75d6d5ca1e9276c03adbfa0357a29442164198ea6931a06efb'
get_class_from_dynamic_module('modeling_minicpmo.MiniCPMO','/projects/u6ada/duplex-live/public-model-slurm-v1/Realtime-Venus-Audio',local_files_only=True)
audio,sr=torchaudio.load('/projects/u6ada/duplex-live/artifacts/native-v75/recovered-data-v1/venus-british-native-v85/references/rp-ben-16k.wav')
assert sr==16000 and audio.ndim==2 and torch.isfinite(audio).all()
(p/'runtime-ready.json').write_text(json.dumps(dict(passed=True,torch=torch.__version__,python=__import__('sys').version,transformers=m.version('transformers'),minicpmo_utils=m.version('minicpmo-utils'),GPU_count=1,model_weights_loaded=False,new_optimizer_updates=0,video_dependencies_unused=True),indent=2)+'\n')
from huggingface_hub import snapshot_download
snapshot=snapshot_download('openai/whisper-large-v3-turbo',revision='41f01f3fe87f28c78e2fbf8b568835947dd65ed9',allow_patterns=['*.json','*.txt','*.safetensors','*.model'],max_workers=4)
(p/'ASR-public-snapshot.json').write_text(json.dumps(dict(path=snapshot,revision='41f01f3fe87f28c78e2fbf8b568835947dd65ed9',public_model_download_only=True,model_execution=False),indent=2)+'\n')
PY
"$PY" -m pip freeze > "$OUT/pip-freeze.txt"
