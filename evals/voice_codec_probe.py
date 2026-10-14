"""Round-trip a synthetic development reference through the native eight-codebook Mimi.

This isolates reconstruction from speech generation. It neither trains a model nor
grants a data-quality approval. Keep the original and reconstructed WAVs audible.
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-run", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import moshi
    from moshi.models import loaders

    expected = args.source / "moshi" / "moshi" / "__init__.py"
    if Path(moshi.__file__).resolve() != expected.resolve():
        raise RuntimeError("Codec probe must use the pinned PersonaPlex implementation")
    environment = json.loads((args.reference_run / "environment.json").read_text())
    config = environment["session_config"]
    if config["model"] != "gpt-live-1" or config["audio"]["output"]["voice"] != "marin":
        raise RuntimeError("Expected the recorded synthetic GPT-Live marin reference")
    original = args.reference_run / "model_out.wav"
    audio, rate = sf.read(original, dtype="float32")
    if rate != 24000 or audio.ndim != 1 or not np.isfinite(audio).all():
        raise RuntimeError("Expected finite mono 24 kHz reference audio")
    # A continuous excerpt of the first general-knowledge answer; no word
    # rearrangement, resynthesis, normalization, or private-result conditioning.
    start, stop = round(3.2 * rate), round(13.2 * rate)
    clip = audio[start:stop]
    if len(clip) != stop-start or float(np.sqrt(np.mean(clip**2))) < .001:
        raise RuntimeError("Missing or silent development reference excerpt")
    args.out.mkdir(parents=True, exist_ok=False)
    sf.write(args.out / "reference.wav", clip, rate, subtype="FLOAT")
    weight = args.model / "tokenizer-e351c8d8-checkpoint125.safetensors"
    mimi = loaders.get_mimi(weight, device="cuda").eval().requires_grad_(False)
    if mimi.sample_rate != rate or mimi.num_codebooks != 8:
        raise RuntimeError("Must use the speech model's actual eight-codebook codec")
    frame = round(rate / mimi.frame_rate)
    padded = np.pad(clip, (0, (-len(clip)) % frame))
    chunks, code_chunks = [], []
    with torch.inference_mode(), mimi.streaming(1):
        for offset in range(0, len(padded), frame):
            x = torch.from_numpy(padded[offset:offset+frame]).reshape(1, 1, -1).cuda()
            codes = mimi.encode(x)
            decoded = mimi.decode(codes)
            if codes.shape[:2] != (1, 8) or decoded.shape[:2] != (1, 1):
                raise RuntimeError("Unexpected native streaming codec shape")
            code_chunks.append(codes.cpu())
            chunks.append(decoded[0, 0].float().cpu().numpy())
    reconstructed = np.concatenate(chunks)
    if len(reconstructed) != len(padded) or not np.isfinite(reconstructed).all():
        raise RuntimeError("Streaming reconstruction changed duration or produced invalid samples")
    reconstructed = reconstructed[:len(clip)]
    sf.write(args.out / "mimi-roundtrip.wav", reconstructed, rate, subtype="FLOAT")
    torch.save(torch.cat(code_chunks, dim=-1), args.out / "codes.pt")
    receipt = {
        "schema": "dl-voice-codec-probe-v1", "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_model": config["model"], "source_voice": "marin",
        "provenance_class": "closed-content", "training_approved": False,
        "derived_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_run": str(args.reference_run), "source_wav_sha256": sha256(original),
        "source_environment_sha256": sha256(args.reference_run / "environment.json"),
        "excerpt_start_s": start/rate, "excerpt_stop_s": stop/rate,
        "codec_weight": str(weight), "codec_sha256": sha256(weight),
        "codec_source_commit": subprocess.check_output(
            ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "num_codebooks": mimi.num_codebooks, "sample_rate": rate,
        "samples": len(clip), "padding_samples": len(padded)-len(clip),
        "reference_sha256": sha256(args.out / "reference.wav"),
        "roundtrip_sha256": sha256(args.out / "mimi-roundtrip.wav"),
        "reconstruction_rms": float(np.sqrt(np.mean(reconstructed**2))),
        "human_quality_verdict": "pending; waveform distance is not a perceptual quality rating",
    }
    (args.out / "report.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
