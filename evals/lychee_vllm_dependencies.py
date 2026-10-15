"""Install an explicit CPU-package subset of the pinned released runtime lock.

Never ask pip to resolve or replace Torch/CUDA/Triton/FlashAttention dependencies.
The original HF voice overlay and compiled backend remain untouched.
"""
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from evals.lychee_model_download import SOURCE_COMMIT
from evals.moshi_model_download import sha256

PACKAGES = set('numpy pyarrow airportsdata astor blake3 cloudpickle compressed-tensors depyf dill diskcache fastapi gguf interegular jsonschema jsonschema-specifications lark llvmlite lm-format-enforcer mistral-common msgspec nest-asyncio numba nvidia-ml-py outlines outlines-core partial-json-parser prometheus-fastapi-instrumentator py-cpuinfo pycountry pydantic-extra-types ray referencing rpds-py starlette tiktoken uvicorn'.split())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('source','target','out'):
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if (subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()!=SOURCE_COMMIT
            or subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True)):
        raise RuntimeError('Runtime lock source changed')
    lock=args.source/'requirements.txt'
    pins=[line for line in lock.read_text().splitlines() if line.split('==')[0] in PACKAGES]
    if {line.split('==')[0] for line in pins}!=PACKAGES:
        raise RuntimeError('Pinned CPU runtime dependencies missing from released lock')
    protected=('torch','torchaudio','torchvision','triton')
    def versions():
        result={}
        for name in protected:
            try: result[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: result[name]=None
        return result
    before=versions()
    args.out.mkdir(exist_ok=False)
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--only-binary=:all:',
        '--disable-pip-version-check','--target',str(args.target),'--report',str(args.out/'pip-report.json'),*pins],check=True)
    if before!=versions():
        raise RuntimeError('Protected container GPU stack changed')
    record={'schema':'dl-lychee-vllm-dependencies-v1','source_commit':SOURCE_COMMIT,
        'lock_sha256':sha256(lock),'pins':pins,'target':str(args.target),'protected_gpu_stack':before,
        'pip_dependency_resolution':False,'original_voice_overlay_changed':False,'passed':True}
    (args.out/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
