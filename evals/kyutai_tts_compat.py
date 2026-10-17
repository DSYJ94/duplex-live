"""Isolated CPU import check for a disclosed streaming-TTS fallback.

Downloads only the hash-pinned pure Python wheel and small pinned metadata.
No weights, GPU allocation, shared environment edits or native quality claim.
"""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import urllib.request
import zipfile

MODEL_REVISION = 'f65439609986c392cb12df63938abcc550c3fb15'
VOICE_REVISION = '323332d33f997de8394f24a193e1a76df720e01a'
WHEEL_SHA256 = 'f426461f1a3fcffd5fb8d317c0a0a482e63766638c89a8ba9c21981b2c88a4f6'


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='dl-kyutai-tts-cpu-compat-v1', passed=False,
        model_revision=MODEL_REVISION, voice_revision=VOICE_REVISION,
        wheel_sha256=WHEEL_SHA256, platform=platform.platform(), python=sys.version,
        shared_environment_modified=False, model_weights_downloaded=False,
        gpu_used=False, native_duplex_qualified=False, hybrid_system_qualified=False)
    try:
        metadata = json.loads(fetch('https://pypi.org/pypi/moshi/0.2.11/json'))
        wheel = next(r for r in metadata['urls'] if r['filename']=='moshi-0.2.11-py3-none-any.whl')
        if wheel['digests']['sha256'] != WHEEL_SHA256:
            raise RuntimeError('Published wheel digest differs from the reviewed pin')
        data = fetch(wheel['url'])
        if hashlib.sha256(data).hexdigest() != WHEEL_SHA256:
            raise RuntimeError('Downloaded wheel digest differs from the reviewed pin')
        (args.out/'moshi-0.2.11.whl').write_bytes(data)
        overlay = args.out/'package'; overlay.mkdir()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                path = PurePosixPath(name)
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError('Unexpected wheel path')
            archive.extractall(overlay)
        config = fetch('https://huggingface.co/kyutai/tts-1.6b-en_fr/resolve/'+MODEL_REVISION+'/config.json')
        (args.out/'config.json').write_bytes(config)
        report.update(config_sha256=hashlib.sha256(config).hexdigest(),
            package_path=str(overlay), declared_dependencies=metadata['info']['requires_dist'])
        sys.path.insert(0, str(overlay))
        import torch
        from moshi.models.loaders import CheckpointInfo
        from moshi.models.tts import TTSModel, script_to_entries
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
            package_versions={n:importlib.metadata.version(n) for n in
                ('moshi','numpy','safetensors','huggingface-hub','sentencepiece','sphn')},
            imported_classes=[CheckpointInfo.__module__, TTSModel.__module__, script_to_entries.__module__],
            torch_within_upstream_requirement=(2,2) <= tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) < (2,8),
            passed=True, meaning='Imports only; dependency version bounds and GPU execution remain separate checks')
    except Exception as error:
        report['error'] = type(error).__name__+': '+str(error)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
