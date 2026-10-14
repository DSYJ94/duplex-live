"""Check every released duplex tensor name and shape on CPU without loading weights."""
import json
import sys
from pathlib import Path
from evals.lychee_environment_check import configure_bf16
from evals.lychee_model_download import SOURCE_COMMIT


def main():
    configure_bf16()
    from accelerate import init_empty_weights
    from safetensors import safe_open
    source = Path("/projects/u6si/duplex-live/vendor") / ("lychee-fd-" + SOURCE_COMMIT)
    model_path = Path("/projects/u6si/duplex-live/models/hf-cache/hub/models--HIT-TMG--Lychee-FD/snapshots/ac06087bef70b33d4dfebf63a3eadadb4e024f33")
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "third_party" / "Step-Audio2"))
    from token2wav import Token2wav
    from Training_package.Code.Models.LycheeFD import LycheeFD, LycheeFDConfig
    config = LycheeFDConfig.from_pretrained(model_path, local_files_only=True)
    with init_empty_weights():
        model = LycheeFD(config)
    expected = {name: list(tensor.shape) for name, tensor in model.state_dict().items()}
    index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    actual = {}
    for filename in sorted(set(index.values())):
        with safe_open(model_path / filename, framework="pt", device="cpu") as stream:
            actual.update({key: list(stream.get_slice(key).get_shape()) for key in stream.keys()})
    if actual != expected:
        raise RuntimeError("Native checkpoint shape/key mismatch: " + repr({
            "missing": sorted(set(expected)-set(actual)), "unexpected": sorted(set(actual)-set(expected)),
            "shapes": [k for k in set(actual)&set(expected) if actual[k] != expected[k]]}))
    print(json.dumps({"checkpoint_tensor_count": len(actual), "all_names_and_shapes_match": True,
                      "weights_loaded": False, "gpu_used": False}), flush=True)


if __name__ == "__main__":
    main()
