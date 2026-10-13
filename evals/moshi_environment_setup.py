"""Plain words: install only a reviewed dependency plan, then test real ARC attention."""
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--requirements", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.verify:
        report = args.out / "dependency-plan.json"
        subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run",
                        "--report", str(report), "-r", str(args.requirements)], check=True)
        plan = json.loads(report.read_text())["install"]
        forbidden = [x["metadata"]["name"] for x in plan
                     if x["metadata"]["name"].lower().replace("_", "-").startswith(
                         ("torch", "nvidia", "triton", "functorch"))]
        if forbidden:
            raise RuntimeError(f"Dependency plan would replace base GPU stack: {forbidden}")
        # Apply the exact versions in the reviewed resolver plan, without another
        # dependency resolution that could replace container Torch/CUDA.
        resolved = [f'{x["metadata"]["name"]}=={x["metadata"]["version"]}' for x in plan]
        if resolved:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", *resolved], check=True)
        return

    import torch
    import moshi
    import transformers
    import sphn
    import gradium
    import xformers
    from moshi.conditioners import arc_encoder
    from xformers.ops import memory_efficient_attention
    from xformers.ops.fmha.attn_bias import BlockDiagonalMask

    result = {"python": sys.version, "torch": torch.__version__,
              "torch_path": torch.__file__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "moshi_path": moshi.__file__,
              "transformers": transformers.__version__, "xformers": xformers.__version__,
              "packages": {}}
    for package in ("sphn", "gradium", "aiohttp", "safetensors", "huggingface-hub"):
        result["packages"][package] = importlib.metadata.version(package)
    if "/envs/" in torch.__file__ or transformers.__version__ != "4.55.3":
        raise RuntimeError("Unexpected base Torch or Transformers")
    torch.manual_seed(424242)
    x = torch.randn(1, 32, 4, 128, device="cuda", dtype=torch.bfloat16)
    bias = BlockDiagonalMask.from_seqlens([16, 16])
    actual = memory_efficient_attention(x, x, x, attn_bias=bias)
    expected = torch.cat([torch.nn.functional.scaled_dot_product_attention(
        part.transpose(1, 2), part.transpose(1, 2), part.transpose(1, 2)
    ).transpose(1, 2) for part in x.split(16, dim=1)], dim=1)
    torch.cuda.synchronize()
    error = float((actual.float() - expected.float()).abs().max().item())
    result["attention_max_abs_error"] = error
    result["arc_kernel_ok"] = bool(torch.isfinite(actual).all().item()) and error < 0.05
    if not result["arc_kernel_ok"]:
        raise RuntimeError(f"ARC attention comparison failed: {result}")
    (args.out / "environment.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "pip-freeze.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
