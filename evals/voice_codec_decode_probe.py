"""Decode the exact captured codec tokens without compilation on CPU or GPU."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codec", type=Path, required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from moshi.models import loaders
    from moshi.utils.compile import no_compile
    receipt = json.loads((args.codec / "report.json").read_text())
    weight = Path(receipt["codec_weight"])
    if sha256(weight) != receipt["codec_sha256"]:
        raise RuntimeError("Native codec weight changed")
    codes = torch.load(args.codec / "codes.pt", map_location="cpu", weights_only=True)
    original, rate = sf.read(args.codec / "mimi-roundtrip.wav", dtype="float32")
    if codes.shape[:2] != (1, 8) or rate != 24000 or not np.isfinite(original).all():
        raise RuntimeError("Invalid captured native codec control")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    mimi = loaders.get_mimi(weight, device=args.device).eval().requires_grad_(False)
    waves = []
    with torch.inference_mode(), no_compile(), mimi.streaming(1):
        for index in range(codes.shape[-1]):
            wave = mimi.decode(codes[..., index:index+1].to(args.device))
            waves.append(wave[0, 0].float().cpu().numpy())
    result = np.concatenate(waves)
    if len(result) != len(original) or not np.isfinite(result).all():
        raise RuntimeError("Decoder control changed sample count or produced invalid speech")
    sf.write(args.out / "decoded.wav", result, rate, subtype="FLOAT")
    error = result-original
    report = {"schema": "dl-codec-decoder-control-v1", "device": args.device,
        "compilation": "explicit native no_compile", "codec_weight_sha256": receipt["codec_sha256"],
        "codes_sha256": sha256(args.codec / "codes.pt"), "samples": len(result),
        "compiled_reference_sha256": sha256(args.codec / "mimi-roundtrip.wav"),
        "decoded_sha256": sha256(args.out / "decoded.wav"),
        "max_abs_difference": float(np.max(np.abs(error))),
        "rms_difference": float(np.sqrt(np.mean(error**2))),
        "human_click_verdict": "pending; numeric difference alone is not voice quality"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
