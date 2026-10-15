"""Execute isolated released-backend CUDA kernels before any model claim."""
import argparse
import importlib
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    import torch
    args.out.mkdir(exist_ok=False)
    build = json.loads((args.build/'build-report.json').read_text())
    rows = build['wheels']
    if len(rows) != 1 or not build['built'] or sha256(args.build/'wheels'/rows[0]['file']) != rows[0]['sha256']:
        raise RuntimeError('Recorded isolated backend wheel changed')
    torch.manual_seed(424242)
    library = args.runtime/'vllm/_C.abi3.so'
    torch.ops.load_library(str(library))
    results = []
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for width in (512, 1024, 3584):
            x = torch.randn(7,width,device='cuda',dtype=dtype)
            weight = torch.randn(width,device='cuda',dtype=dtype)
            actual = torch.empty_like(x)
            torch.ops._C.rms_norm(actual,x,weight,1e-6)
            reference = (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(dtype)*weight
            limit = (2*torch.finfo(dtype).eps*reference.float().abs().clamp_min(1))
            error = (actual.float()-reference.float()).abs()
            passed = bool(torch.isfinite(actual).all() and (error <= limit).all())
            results.append({'operation':'rms_norm','dtype':str(dtype),'width':width,
                'maximum_error':float(error.max()),'passed':passed})
            if not passed:
                raise RuntimeError('Native CUDA13 RMS kernel differs from FP32 reduction reference: '+str(results[-1]))
    record = {'schema':'dl-lychee-vllm-kernel-v1','torch':torch.__version__, 'cuda':torch.version.cuda,
        'library_sha256':sha256(library),'wheel_sha256':rows[0]['sha256'],'probes':results,
        'passed':True,'full_model_loaded':False,'audio_quality_or_realtime_qualified':False}
    (args.out/'kernel-report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    # A successful library import alone is insufficient. Check the patched
    # Python package separately, allowing missing dependencies to fail visibly.
    vllm = importlib.import_module('vllm')
    if not Path(vllm.__file__).resolve().is_relative_to(args.runtime.resolve()):
        raise RuntimeError('Imported a different vLLM implementation')
    if not hasattr(vllm,'MultiHeadRequestState'):
        raise RuntimeError('Released custom multi-head runtime is absent')
    (args.out/'python-import.json').write_text(json.dumps({'passed':True,'file':vllm.__file__,
        'version':vllm.__version__,'model_load_tested':False},indent=2)+'\n')


if __name__ == '__main__':
    main()
