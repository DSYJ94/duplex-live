"""Recorded CCCL 3.0 operation-name migration in an isolated CUDA13 build only.

NVIDIA migration source:
https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html
Weights, model architecture and accepted vendor checkouts are untouched.
Numerical native-kernel tests are still required after a successful build.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

FILES = {
    "csrc/layernorm_kernels.cu": "57d5d4e2912e0661d57a0eab4a837ea7a8cab794b7e4cf10b7d5f42dd31ef2d8",
    "csrc/layernorm_quant_kernels.cu": "e379cdd478f9ae5b2e8bf61aec288b7b0bb76b341a9bc67565ca03e01adde02c",
    "csrc/moe/topk_softmax_kernels.cu": "6a85f9f6ebc7e228eb50fbb2d0272d54de42c49acff4f21bcaa794c14c3f4482",
    "csrc/quantization/compressed_tensors/int8_quant_kernels.cu": "2f9732980a4a6c1752d532e83c97b3373ae9352f25c48ab01d37c638d6c831e7",
    "csrc/quantization/fp8/common.cu": "8075d34952215fab2b80461d28108743e5e819413b7e3a29292ce77a9c31bf61",
    "csrc/quantization/fused_kernels/layernorm_utils.cuh": "f05fb9f0d8d6535062915b7f57fe55c307b43bc263b082565d8e0a27b09f0f51",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not args.source.name.startswith("lychee-vllm-source-") or subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip() != "7bb1068c59b578e1f8e65fb762cd26708c14cc6d":
        raise RuntimeError("Compatibility changes require a separate pinned build checkout")
    base = args.source / "third_party/vllm"
    digest = lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    if any(digest(base / name) != expected for name,expected in FILES.items()):
        raise RuntimeError("Original CUDA reduction sources differ from the pinned release")
    substitutions = {"Sum":"cuda::std::plus<>", "Max":"cuda::maximum<>", "Min":"cuda::minimum<>"}
    reports = []
    for name, before in FILES.items():
        path = base / name
        text = path.read_text()
        text, count = re.subn(r"\bcub::(Sum|Max|Min)\b", lambda m:substitutions[m[1]], text)
        if not count or "#include" not in text:
            raise RuntimeError("Expected pinned reduction operations were not found")
        text = text.replace("#include", "#include <cuda/std/functional>\n#include <cuda/functional>\n\n#include", 1)
        path.write_text(text)
        reports.append({"file":name, "before_sha256":before, "after_sha256":digest(path), "substitutions":count})
    diff = subprocess.check_output(["git", "-C", str(args.source), "diff", "--binary"])
    args.out.write_text(json.dumps({"schema":"dl-lychee-cuda13-compat-v1", "scope":"isolated CUDA13 build only",
        "source_commit":"7bb1068c59b578e1f8e65fb762cd26708c14cc6d", "files":reports,
        "diff_sha256":hashlib.sha256(diff).hexdigest(), "native_numerical_tests_passed":False}, indent=2)+"\n")
    print(json.dumps({"compatibility_files":len(reports), "diff_sha256":hashlib.sha256(diff).hexdigest()}), flush=True)


if __name__ == "__main__":
    main()
