"""Audible native PersonaPlex comparison on approved development user audio.

Offline replay establishes content and voice only, not wall-clock live latency.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from evals.moshi_model_download import sha256


def main():
    import numpy as np
    import soundfile as sf
    import torch
    import moshi
    from scipy.signal import resample_poly
    from moshi.offline import run_inference
    from data.synthetic.quality import checked_rows

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--voice-provenance", type=Path,
                        help="Required codec receipt when conditioning on a synthetic reference WAV")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    voice_provenance = None
    if args.voice.suffix.lower() != ".pt":
        if args.voice.suffix.lower() != ".wav" or not args.voice_provenance:
            parser.error("Custom WAV conditioning requires --voice-provenance")
        voice_provenance = json.loads(args.voice_provenance.read_text())
        if (voice_provenance.get("schema") != "dl-voice-codec-probe-v1"
                or voice_provenance.get("reference_sha256") != sha256(args.voice)
                or voice_provenance.get("source_model") != "gpt-live-1"):
            raise RuntimeError("Custom synthetic voice reference does not match its receipt")
    scenario = "dense_dialog-001"
    checked_rows(args.batch, [{"scenario_id": scenario}], role="PersonaPlex development comparison")
    expected = args.source / "moshi" / "moshi" / "__init__.py"
    if Path(moshi.__file__).resolve() != expected.resolve():
        raise RuntimeError("Wrong Moshi module: must use the pinned NVIDIA source")
    if subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True):
        raise RuntimeError("PersonaPlex source is dirty")
    assets = [args.model / name for name in (
        "model.safetensors", "tokenizer-e351c8d8-checkpoint125.safetensors",
        "tokenizer_spm_32k_3.model", "config.json")] + [args.voice]
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_LIBCUDA_PATH"] = subprocess.check_output([
        sys.executable, "-m", "evals.cuda_linker_path", "--out", str(args.out / "driver")], text=True).strip()
    env = {"model_revision": args.model.name, "source_commit": subprocess.check_output(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "torch_file": torch.__file__, "cuda": torch.version.cuda,
        "input_scenario": scenario, "fresh_holdout": False, "paced": False,
        "microphone_acceptance": False, "input_seconds": 60, "added_silence_seconds": 10,
        "assets": {str(p): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in assets}}
    env["voice_conditioning"] = {"path": str(args.voice), "provenance": voice_provenance,
                                 "training_approved": False}
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    pcm, sr = sf.read(args.batch / scenario / "stem_user.wav", dtype="float32")
    if pcm.ndim != 1 or not np.isfinite(pcm).all() or len(pcm) < 60 * sr:
        raise RuntimeError("Invalid or too short verified input")
    pcm = pcm[:60 * sr]
    divisor = math.gcd(sr, 24000)
    pcm = resample_poly(pcm, 24000 // divisor, sr // divisor)
    pcm = np.concatenate([pcm, np.zeros(10 * 24000, dtype=np.float32)])
    trial = args.out / "t1"
    trial.mkdir()
    sf.write(trial / "user_in.wav", pcm, 24000)
    started = time.monotonic()
    run_inference(input_wav=str(trial / "user_in.wav"), output_wav=str(trial / "model_out.wav"),
        output_text=str(trial / "output.json"),
        text_prompt="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
        voice_prompt_path=str(args.voice), tokenizer_path=str(assets[2]),
        moshi_weight=str(assets[0]), mimi_weight=str(assets[1]),
        hf_repo="nvidia/personaplex-7b-v1", device="cuda", seed=42424242,
        temp_audio=0.8, temp_text=0.7, topk_audio=250, topk_text=25,
        greedy=False, save_voice_prompt_embeddings=False)
    audio, rate = sf.read(trial / "model_out.wav")
    if not np.isfinite(audio).all() or not np.any(np.abs(audio) > 0.001):
        raise RuntimeError("Empty or invalid generated speech")
    tokens = json.loads((trial / "output.json").read_text())
    summary = {"wall_seconds_including_load": time.monotonic() - started,
        "output_seconds": len(audio) / rate,
        "generated_text": "".join(t for t in tokens if t not in ("EPAD", "PAD", "BOS", "EOS")),
        "voice_quality_verdict": "human audition required", "live_acceptance": False}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
