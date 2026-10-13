"""Plain words: check the clean container's CUDA/attention compatibility.

Run without an overlay first. This does not install or replace any packages.
"""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    result = {"python": sys.version, "machine": platform.machine(),
              "torch": torch.__version__, "torch_path": torch.__file__,
              "cuda": torch.version.cuda, "packages": {}}
    for package in ("transformers", "xformers", "torchaudio", "torchvision",
                    "sentencepiece", "sphn", "huggingface-hub", "peft"):
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    # CUDA is a hard requirement; missing optional dependencies are a finding.
    x = torch.randn(1, 32, 4, 64, device="cuda", dtype=torch.bfloat16)
    result["gpu"] = torch.cuda.get_device_name()
    result["cuda_operation_ok"] = bool(torch.isfinite(x @ x.transpose(-1, -2)).all().item())
    try:
        from xformers.ops import memory_efficient_attention
        from xformers.ops.fmha.attn_bias import BlockDiagonalMask
        bias = BlockDiagonalMask.from_seqlens([16, 16])
        y = memory_efficient_attention(x, x, x, attn_bias=bias)
        torch.cuda.synchronize()
        result["xformers_arc_shape_ok"] = bool(torch.isfinite(y).all().item())
    except Exception as error:
        result["xformers_arc_shape_ok"] = False
        result["xformers_error"] = f"{type(error).__name__}: {error}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
