"""Plain words: transcribe generated native speech and retain the actual audio evidence.

ASR agreement is an intelligibility diagnostic, not a semantic or human rating.
This does not grant a training-data approval stamp to generated model output.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words, wer
    args.out.mkdir(parents=True, exist_ok=False)
    model = whisper.load_model("turbo", device="cuda")
    results = []
    for run in args.runs:
        wav = run / "model.wav"
        pcm, sr = sf.read(wav, dtype="float32")
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("Invalid generated mono speech")
        g = math.gcd(sr, 16000)
        audio = resample_poly(pcm, 16000//g, sr//g).astype(np.float32)
        result = whisper.transcribe(model, audio, language="en", vad=False,
                                    beam_size=5, best_of=5, temperature=0., verbose=None)
        predicted = json.loads((run / "report.json").read_text())["text"]
        record = {"run": str(run), "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                  "audio_duration_s": len(pcm)/sr, "model_text": predicted,
                  "asr_text": result["text"], "asr_segments": result["segments"],
                  "wer_against_model_text": wer(norm_words(predicted), norm_words(result["text"])),
                  "semantic_review": "pending", "human_listening": "pending"}
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "asr_segments"}), flush=True)
    (args.out / "report.json").write_text(json.dumps({"schema": "dl-native-speech-audit-v1",
        "whisper": importlib.metadata.version("openai-whisper"),
        "timestamped": importlib.metadata.version("whisper-timestamped"), "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
