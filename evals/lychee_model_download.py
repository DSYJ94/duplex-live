"""Fetch pinned Lychee-FD evaluation assets without changing working environments."""
import argparse
import json
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256

SOURCE_COMMIT = "7bb1068c59b578e1f8e65fb762cd26708c14cc6d"
MODELS = {
    "duplex": ("HIT-TMG/Lychee-FD", "ac06087bef70b33d4dfebf63a3eadadb4e024f33", None),
    "vocoder": ("stepfun-ai/Step-Audio-2-mini", "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36", ["token2wav/*"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vendor", type=Path, required=True)
    args = ap.parse_args()
    from huggingface_hub import snapshot_download
    args.out.mkdir(parents=True, exist_ok=False)
    if not args.vendor.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/HITsz-TMG/Lychee-FD.git", str(args.vendor)], check=True)
        subprocess.run(["git", "-C", str(args.vendor), "sparse-checkout", "set",
                        "Training_package", "lychee_fd", "third_party/Step-Audio2"], check=True)
        subprocess.run(["git", "-C", str(args.vendor), "checkout", "--detach", SOURCE_COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(args.vendor), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(args.vendor), "status", "--porcelain"], text=True)
    if actual != SOURCE_COMMIT or dirty:
        raise RuntimeError("Lychee source must be clean and pinned")
    receipt = {"schema": "dl-lychee-assets-v1", "source": str(args.vendor),
               "source_commit": actual, "models": {}, "qualification": "assets only; no inference result"}
    for label, (repo, revision, patterns) in MODELS.items():
        print(json.dumps({"event": "download_start", "repo": repo, "revision": revision}), flush=True)
        path = Path(snapshot_download(repo, revision=revision, allow_patterns=patterns, max_workers=2))
        receipt["models"][label] = {"repo": repo, "revision": revision, "path": str(path),
            "files": {str(p.relative_to(path)): {"sha256": sha256(p), "bytes": p.stat().st_size}
                      for p in path.rglob("*") if p.is_file()}}
        (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
        print(json.dumps({"event": "download_complete", "label": label, "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
