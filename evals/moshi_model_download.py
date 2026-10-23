"""Plain words: fetch exact public speech-model revisions and record file hashes."""
import argparse
import hashlib
import json
from pathlib import Path


MODELS = {
    "rag": ("kyutai/moshika-rag-pytorch-bf16", "7135a6e3c46abb66c2cd95cb04cbfcbe8376f83d"),
    "arc": ("kyutai/ARC4_Encoder_Llama", "c11e53d1016cc586262ee883755410e2ca47ba3c"),
    "stt": ("kyutai/stt-1b-en_fr-candle", "095e38f6242006a93c2541149b181988397f5c7c"),
    "native": ("kyutai/moshiko-pytorch-bf16", "2bfc9ae6e89079a5cc7ed2a68436010d91a3d289"),
}


def sha256(path):
    from evals.immutable_file_hashes import active
    session=active()
    return session.digest(path,_sha256_uncached) if session is not None else _sha256_uncached(path)


def _sha256_uncached(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for label, (repo, revision) in MODELS.items():
        print(json.dumps({"event": "download_start", "repo": repo, "revision": revision}), flush=True)
        path = Path(snapshot_download(repo, revision=revision, max_workers=2,
                                     allow_patterns=["*.json", "*.model", "*.safetensors"]))
        manifest[label] = {"repo": repo, "revision": revision, "path": str(path),
                           "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                                     for p in path.iterdir() if p.is_file()}}
        (args.out / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({"event": "download_complete", "label": label, "path": str(path)}), flush=True)
    # ARC uses the official Llama tokenizer. Test actual access; metadata alone
    # does not prove gated files are available. Never print tokens or auth headers.
    repo = "meta-llama/Llama-3.2-3B-Instruct"
    revision = HfApi().model_info(repo).sha
    try:
        path = Path(snapshot_download(repo, revision=revision, max_workers=2,
                                     allow_patterns=["tokenizer*", "special_tokens_map.json"]))
        manifest["arc_tokenizer"] = {"repo": repo, "revision": revision, "path": str(path),
                                     "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                                               for p in path.iterdir() if p.is_file()}}
    except Exception as error:
        manifest["arc_tokenizer"] = {"repo": repo, "revision": revision,
                                     "available": False, "error_type": type(error).__name__}
        print(json.dumps({"event": "tokenizer_access_failed", "type": type(error).__name__}), flush=True)
    (args.out / "model-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
