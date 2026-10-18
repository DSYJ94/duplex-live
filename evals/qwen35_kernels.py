"""Separate byte-verified optional FLA kernel assets; no shared installation.

This CPU stage proves wheel/package provenance only. CUDA import, compilation,
selected model functions, numerical behavior and latency require a later GPU
proof. The depthwise convolution remains the existing PyTorch implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import zipfile
from evals.moshi_model_download import sha256
from evals.qwen35_runtime import verified_runtime


def inventory(package):
    return {str(p.relative_to(package)):sha256(p) for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def verified_kernels(path,runtime):
    report=json.loads((path/'report.json').read_text());pins=Path(__file__).with_name('assets')/'qwen35-kernel-pins.json'
    if (report['schema']!='dl-qwen35-kernel-assets-v1' or not report['passed']
            or report['code_sha256']!=sha256(Path(__file__)) or report['pins_sha256']!=sha256(pins)
            or report['runtime_report_sha256']!=sha256(runtime/'report.json')
            or inventory(path/'package')!=report['package_files']):raise RuntimeError('Missing exact optional kernel package proof')
    return path/'package',report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    runtime,_=verified_runtime(args.runtime);args.out.mkdir(parents=True,exist_ok=False)
    pins_path=Path(__file__).with_name('assets')/'qwen35-kernel-pins.json';pins=json.loads(pins_path.read_text());wheel_dir=args.out/'wheels';wheel_dir.mkdir();package=args.out/'package';wheels=[];expected={}
    for row in pins['files']:
        target=wheel_dir/row['filename']
        with urllib.request.urlopen(row['url'],timeout=90) as response:target.write_bytes(response.read())
        if target.stat().st_size!=row['bytes'] or sha256(target)!=row['sha256']:raise RuntimeError('Changed optional kernel wheel')
        with zipfile.ZipFile(target) as archive:
            for name in archive.namelist():
                if not name.startswith('fla/') or name.endswith('/'):continue
                digest=hashlib.sha256(archive.read(name)).hexdigest()
                if name in expected and expected[name]!=digest:raise RuntimeError('Conflicting FLA wheel module contents')
                expected[name]=digest
        wheels.append(target)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(package),*[str(p) for p in wheels]],check=True)
    for name,digest in expected.items():
        if not (package/name).is_file() or sha256(package/name)!=digest:raise RuntimeError('Kernel code differs from exact wheel')
    sys.path.insert(0,str(runtime));sys.path.insert(0,str(package))
    import importlib.metadata,torch,triton,einops
    versions={name:importlib.metadata.version(name) for name in ('flash-linear-attention','fla-core','transformers')}
    if versions['flash-linear-attention']!='0.5.2' or versions['fla-core']!='0.5.2' or versions['transformers']!='5.5.3':raise RuntimeError('Kernel/runtime distribution metadata mismatch')
    report=dict(schema='dl-qwen35-kernel-assets-v1',passed=True,package_files=inventory(package),
        wheel_code_files_exact=len(expected),versions=dict(versions,torch=torch.__version__,triton=triton.__version__,einops=einops.__version__),
        pins_sha256=sha256(pins_path),code_sha256=sha256(Path(__file__)),runtime_report_sha256=sha256(args.runtime/'report.json'),
        package_assets_only=True,fla_gpu_import_executed=False,model_inference_executed=False,gpu_used=False,
        causal_conv1d_installed=False,shared_packages_changed=False,training_approved=False)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='package_files'}),flush=True)


if __name__=='__main__':main()
