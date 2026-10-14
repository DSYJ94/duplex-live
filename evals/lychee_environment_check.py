"""Explicit BF16-only compatibility configuration for the isolated Lychee probe."""


def configure_bf16():
    import transformers
    import transformers.utils.import_utils as imports
    if transformers.__version__ != "4.57.6" or not hasattr(imports, "_torchao_available"):
        raise RuntimeError("Unexpected HF version for the recorded BF16 compatibility rule")
    # NGC's installed experimental TorchAO lacks the API HF imports. This
    # evaluation loads ordinary BF16 parameters; no quantization is requested.
    # Disable only HF's discovery of that unused optional quantization backend.
    imports._torchao_available = False
    return {"transformers": transformers.__version__, "torchao_discovery": "disabled explicitly; BF16 only"}


def main():
    import json
    rule = configure_bf16()
    import torch
    import torchaudio
    import peft
    import datasets
    import librosa
    import s3tokenizer
    import onnxruntime
    import hyperpyyaml
    print(json.dumps({**rule, "torch": torch.__version__, "torch_file": torch.__file__,
                      "torchaudio": torchaudio.__version__, "imports_passed": True}), flush=True)


if __name__ == "__main__":
    main()
