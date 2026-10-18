"""Isolated hash-pinned ARM64 Transformers runtime; shared packages untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
from evals.moshi_model_download import sha256

PACKAGES=('transformers','huggingface_hub','tokenizers','safetensors','regex')


def verified_runtime(path):
    report=json.loads((path/'report.json').read_text());pin=Path(__file__).with_name('assets')/'qwen35-runtime-pins.json'
    if (report['schema']!='dl-qwen35-isolated-runtime-v1' or not report['passed']
            or report['code_sha256']!=sha256(Path(__file__)) or report['pins_sha256']!=sha256(pin)):
        raise RuntimeError('Missing exact isolated runtime proof')
    package=path/'package';actual={str(p.relative_to(package)):sha256(p) for name in PACKAGES for p in (package/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    if actual!=report['package_files']:raise RuntimeError('Changed isolated runtime bytes')
    return package,report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);wheel_dir=args.out/'wheels';wheel_dir.mkdir();package=args.out/'package'
    pin=Path(__file__).with_name('assets')/'qwen35-runtime-pins.json';pins=json.loads(pin.read_text());wheels=[]
    for row in pins['files']:
        target=wheel_dir/row['filename']
        with urllib.request.urlopen(row['url'],timeout=90) as response,target.open('wb') as sink:
            while True:
                data=response.read(1024*1024)
                if not data:break
                sink.write(data)
        if target.stat().st_size!=row['bytes'] or sha256(target)!=row['sha256']:raise RuntimeError('Changed pinned wheel')
        wheels.append(target)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(package),*[str(p) for p in wheels]],check=True)
    checked=0
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith('/') or name.split('/')[0] not in PACKAGES:continue
                installed=package/name
                if not installed.is_file() or hashlib.sha256(archive.read(name)).hexdigest()!=sha256(installed):
                    raise RuntimeError('Installed package differs from original wheel: '+name)
                checked+=1
    sys.path.insert(0,str(package))
    import torch,transformers,huggingface_hub,tokenizers,safetensors
    from transformers import AutoConfig,AutoTokenizer,Qwen3_5ForConditionalGeneration
    from transformers.modeling_utils import no_init_weights
    if transformers.__version__!='5.5.3':raise RuntimeError('Wrong isolated Transformers import')
    files={str(p.relative_to(package)):sha256(p) for name in PACKAGES for p in (package/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    report=dict(schema='dl-qwen35-isolated-runtime-v1',passed=True,package=str(package),package_files=files,
        wheel_code_files_exact=checked,versions=dict(torch=torch.__version__,transformers=transformers.__version__,
            huggingface_hub=huggingface_hub.__version__,tokenizers=tokenizers.__version__,safetensors=safetensors.__version__),
        actual_cpu_model_class_import=Qwen3_5ForConditionalGeneration.__name__,no_init_weights_imported=True,
        pins_sha256=sha256(pin),code_sha256=sha256(Path(__file__)),shared_packages_changed=False,
        model_inference_executed=False,gpu_used=False,training_approved=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='package_files'}),flush=True)


if __name__=='__main__':main()
