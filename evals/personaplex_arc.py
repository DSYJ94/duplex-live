"""Load the pinned frozen ARC encoder alongside NVIDIA's separate Moshi package.

Use a distinct package namespace; never replace the foreground's moshi imports.
"""
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from evals.moshi_model_download import sha256


def load_encoder(source, manifest_path, tokenizer_path, out):
    import torch
    from safetensors import safe_open
    import moshi
    foreground = moshi
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != "8c6dfc101b7871baa428424bcdc583b74fb561d9":
        raise RuntimeError("Unpinned ARC source")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True):
        raise RuntimeError("ARC source is dirty")
    namespace = "_duplex_frozen_arc"
    root = source / "moshi" / "moshi"
    spec = importlib.util.spec_from_file_location(namespace, root / "__init__.py", submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[namespace] = module
    spec.loader.exec_module(module)
    upstream = importlib.import_module(namespace + ".server_conditioner")
    if sys.modules["moshi"] is not foreground:
        raise RuntimeError("ARC import replaced foreground Moshi")
    manifest = json.loads(manifest_path.read_text())
    for label in ("rag", "arc"):
        if "model.safetensors" not in manifest[label]["files"]:
            raise RuntimeError("Encoder manifest omits required weights")
        for name, record in manifest[label]["files"].items():
            if name not in ("config.json", "model.safetensors"):
                continue
            if sha256(Path(manifest[label]["path"]) / name) != record["sha256"]:
                raise RuntimeError("Pinned encoder asset changed")
    config = json.loads((Path(manifest["rag"]["path"]) / "config.json").read_text())
    config["conditioners"]["reference_with_time"]["multi_arc_encoder"].update(tokenizer_name=str(tokenizer_path), hf_repo=None)
    config_path = out / "arc-config.json"
    config_path.write_text(json.dumps(config, indent=2)+"\n")

    class AuditedEncoder(upstream.EncoderService):
        def _load_conditioner_weights_from_moshi_checkpoint(self, checkpoint):
            prefix = "condition_provider.conditioners.reference_with_time."
            with safe_open(checkpoint, framework="pt", device="cpu") as file:
                state = {k.removeprefix(prefix): file.get_tensor(k) for k in file.keys() if k.startswith(prefix)}
            with safe_open(str(Path(manifest["arc"]["path"]) / "model.safetensors"), framework="pt", device="cpu") as file:
                state.update({k: file.get_tensor(k) for k in file.keys()})
            conditioner = self.condition_provider.conditioners[self.conditioner_name]
            expected = set(conditioner.state_dict())
            unused = {"embedder.cont_tok.0.weight", "embedder.rec_tok.0.weight"}
            if set(state)-expected != unused:
                raise RuntimeError("Unexpected frozen encoder checkpoint extras")
            for key in unused:
                del state[key]
            if set(state) != expected:
                raise RuntimeError("Incomplete encoder checkpoint")
            conditioner.load_state_dict(state, strict=True)
            conditioner.eval().requires_grad_(False)

    with torch.inference_mode():
        encoder = AuditedEncoder(str(config_path), str(Path(manifest["rag"]["path"]) / "model.safetensors"), "reference_with_time", "cuda:0")
        warm = encoder.encode("Background reference warmup.")
    if not torch.isfinite(warm).all() or warm.shape[-1] != 4096:
        raise RuntimeError("Invalid frozen encoder warmup")
    return encoder
