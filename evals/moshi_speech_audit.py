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
    ap.add_argument("--runs", nargs="+", type=Path, default=[])
    ap.add_argument("--live-runs", nargs="+", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trim-trailing-silence", action="store_true")
    args = ap.parse_args()
    if not args.runs and not args.live_runs:
        ap.error("At least one native or GPT-Live run is required")
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words, wer
    args.out.mkdir(parents=True, exist_ok=False)
    model = whisper.load_model("turbo", device="cuda")
    results = []
    for run,kind in [(r,"native") for r in args.runs]+[(r,"GPT-Live") for r in args.live_runs]:
        wav = run / ("model.wav" if kind == "native" else "model_out.wav")
        pcm, sr = sf.read(wav, dtype="float32")
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("Invalid generated mono speech")
        g = math.gcd(sr, 16000)
        audio = resample_poly(pcm, 16000//g, sr//g).astype(np.float32)
        original_samples = len(audio)
        if args.trim_trailing_silence:
            # Keep the original WAV intact. Whisper can hallucinate repeated
            # words over a long quiet tail; trim only that tail for this
            # diagnostic, retaining 250 ms after the last active 20 ms window.
            frame = 320
            active = [i for i in range(0, len(audio), frame)
                      if float(np.sqrt(np.mean(audio[i:i+frame] ** 2))) > 0.001]
            if not active:
                raise RuntimeError("Generated audio has no signal above the diagnostic silence floor")
            audio = audio[:min(len(audio), active[-1] + frame + 4000)]
        result = whisper.transcribe(model, audio, language="en", vad=False,
                                    beam_size=5, best_of=5, temperature=0., verbose=None)
        if kind == "native":
            predicted = json.loads((run / "report.json").read_text())["text"]
        else:
            predicted = "".join(p["text"] for p in json.loads((run / "transcripts.json").read_text()) if p["speaker"] == "assistant")
        record = {"run": str(run), "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                  "audio_duration_s": len(pcm)/sr, "model_text": predicted,
                  "asr_input_duration_s": len(audio)/16000,
                  "trailing_silence_trimmed_s": (original_samples-len(audio))/16000,
                  "silence_trim_enabled": args.trim_trailing_silence,
                  "asr_text": result["text"], "asr_segments": result["segments"],
                  "wer_against_model_text": wer(norm_words(predicted), norm_words(result["text"])),
                  "semantic_review": "pending", "human_listening": "pending"}
        record["source_kind"] = kind
        record["asr_time_bounds_valid"] = all(
            0 <= segment["start"] < segment["end"] <= len(audio)/16000 + 0.04
            for segment in result["segments"])
        record["asr_word_bounds_valid"] = all(0 <= word["start"] < word["end"] <= len(audio)/16000+.04
            for segment in result["segments"] for word in segment.get("words",[]))
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "asr_segments"}), flush=True)
    (args.out / "report.json").write_text(json.dumps({"schema": "dl-native-speech-audit-v1",
        "whisper": importlib.metadata.version("openai-whisper"),
        "timestamped": importlib.metadata.version("whisper-timestamped"), "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
