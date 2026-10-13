"""Plain words: replay verified audio through native duplex speech generation.

Records the actual time each output packet becomes available. No scripted
answers, VAD turn controller, timing checkpoint, or gold assistant words enter
the model. This is paced replay, not a microphone/network acceptance test.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--scenario", default="dense_dialog-001")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if not 1 <= args.seconds <= 600:
        ap.error("--seconds must be between 1 and 600")
    from data.synthetic.quality import checked_rows
    checked_rows(args.batch, [{"scenario_id": args.scenario}], role="native duplex replay")
    import numpy as np
    import soundfile as sf
    import torch
    import transformers
    from scipy.signal import resample_poly
    from transformers import AutoModel

    args.out.mkdir(parents=True, exist_ok=False)
    def read_audio(path, rate):
        x, sr = sf.read(path, dtype="float32")
        if x.ndim != 1:
            raise ValueError("Expected mono audio")
        if sr != rate:
            g = math.gcd(sr, rate)
            x = resample_poly(x, rate // g, sr // g).astype(np.float32)
        return x

    pcm = read_audio(args.batch / args.scenario / "stem_user.wav", 16000)
    # Never pad a short fixture into a purported long-session evaluation.
    if len(pcm) < args.seconds * 16000:
        raise ValueError("Verified fixture is shorter than requested replay")
    pcm = pcm[:args.seconds * 16000]
    torch.manual_seed(424242)
    env = {"mode": "paced-native-duplex-replay", "torch": torch.__version__,
           "torch_path": torch.__file__, "transformers": transformers.__version__,
           "model_snapshot": str(args.model), "gpu": torch.cuda.get_device_name(),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "seconds": args.seconds, "scenario": args.scenario,
           "semantic_verdict": "pending review", "microphone_acceptance": False}
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    model = AutoModel.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True,
        attn_implementation="sdpa", torch_dtype=torch.bfloat16,
        init_vision=False, init_audio=True, init_tts=True).eval().cuda()
    model.init_tts()
    duplex = model.as_duplex()
    voice = args.model / "assets" / "HT_ref_audio.wav"
    duplex.prepare(prefix_system_prompt="Streaming Omni Conversation. Respond briefly in English.",
                   ref_audio=read_audio(voice, 16000), prompt_wav_path=str(voice))
    sf.write(args.out / "stem_user.wav", pcm, 16000)
    output = np.zeros(0, dtype=np.float32)
    records = []
    torch.cuda.synchronize()
    start = time.monotonic()
    with (args.out / "packets.jsonl").open("w") as log:
        for c in range(args.seconds):
            # Chunk c is available only after second c+1, never beforehand.
            due = c + 1
            time.sleep(max(0, start + due - time.monotonic()))
            before = time.monotonic()
            with torch.inference_mode():
                pre = duplex.streaming_prefill(audio_waveform=pcm[c*16000:(c+1)*16000])
                result = duplex.streaming_generate(prompt_wav_path=str(voice))
            torch.cuda.synchronize()
            completed = time.monotonic()
            audio = result.get("audio_waveform")
            samples = 0
            playback_start = None
            if audio is not None:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().float().cpu().numpy()
                audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                if not np.isfinite(audio).all():
                    raise ValueError("Nonfinite speech packet")
                samples = len(audio)
                # Model output cannot play before its packet is generated.
                first_sample = max(len(output), math.ceil((completed-start)*24000))
                if samples:
                    playback_start = first_sample / 24000
                    output = np.pad(output, (0, first_sample-len(output)))
                    output = np.concatenate([output, audio])
            rec = {"chunk": c, "input_available_s": due,
                   "input_consumed_s": before-start, "packet_available_s": completed-start,
                   "compute_s": completed-before, "input_backlog_s": before-start-due,
                   "text": result.get("text", ""), "is_listen": result.get("is_listen"),
                   "end_of_turn": result.get("end_of_turn"), "samples": samples,
                   "playback_start_s": playback_start, "prefill_ok": pre.get("success")}
            records.append(rec)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
    if len(output):
        sf.write(args.out / "model.wav", output, 24000)
    report = {"schema": "dl-native-duplex-replay-v1", "seconds": args.seconds,
              "max_input_backlog_s": max(r["input_backlog_s"] for r in records),
              "final_input_backlog_s": records[-1]["input_backlog_s"],
              "compute_p50_s": float(np.median([r["compute_s"] for r in records])),
              "compute_p95_s": float(np.percentile([r["compute_s"] for r in records], 95)),
              "text": " ".join(r["text"] or "" for r in records),
              "semantic_verdict": "pending review", "human_listening": "pending"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "manifest.jsonl").write_text(json.dumps({
        "scenario_id": args.scenario, "dir": ".", "scenario_class": "native-duplex-replay",
        "duration_ms": args.seconds*1000}) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
