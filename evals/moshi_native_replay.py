"""Plain words: measure native generated speech from user-only paced audio."""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-manifest", type=Path, required=True)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=int, default=40)
    args = ap.parse_args()
    from data.synthetic.quality import checked_rows
    checked_rows(args.batch, [{"scenario_id": args.scenario}], role="native Moshi paced replay")
    import numpy as np
    import soundfile as sf
    import torch
    from moshi.models import loaders, LMGen
    from scipy.signal import resample_poly
    from safetensors.torch import load_file

    args.out.mkdir(parents=True, exist_ok=False)
    model_dir = Path(json.loads(args.model_manifest.read_text())["native"]["path"])
    cp = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=None, moshi_weights=model_dir / "model.safetensors",
        mimi_weights=model_dir / "tokenizer-e351c8d8-checkpoint125.safetensors",
        tokenizer=model_dir / "tokenizer_spm_32k_3.model", config_path=args.config,
        lora_weights=args.adapter)
    torch.manual_seed(424242)
    mimi = cp.get_mimi(device="cuda").eval()
    lm = cp.get_moshi(device="cuda", dtype=torch.bfloat16, fuse_lora=False).eval()
    if args.adapter:
        expected = {n for n, _ in lm.named_parameters() if ".lora_A." in n or ".lora_B." in n}
        if set(load_file(args.adapter)) != expected:
            raise RuntimeError("Incomplete adapter in speech replay")
    tokenizer = cp.get_text_tokenizer()
    generation = LMGen(lm, **cp.lm_gen_config)
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    pcm, sr = sf.read(args.batch / args.scenario / "stem_user.wav", dtype="float32")
    if pcm.ndim != 1:
        raise ValueError("Expected verified mono user stem")
    if sr != mimi.sample_rate:
        g = math.gcd(sr, mimi.sample_rate)
        pcm = resample_poly(pcm, mimi.sample_rate//g, sr//g).astype(np.float32)
    if len(pcm) < args.seconds * mimi.sample_rate:
        raise ValueError("Requested replay exceeds verified input")
    pcm = pcm[:args.seconds * mimi.sample_rate]
    sf.write(args.out / "stem_user.wav", pcm, mimi.sample_rate)
    env = {"mode": "native-Moshi-paced-replay", "module": loaders.__file__,
           "torch": torch.__version__, "config": str(args.config), "adapter": str(args.adapter),
           "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
           "seconds": args.seconds, "seed": 424242, "microphone_acceptance": False}
    (args.out / "environment.json").write_text(json.dumps(env, indent=2)+"\n")
    records = []
    output = np.zeros(0, dtype=np.float32)
    with mimi.streaming(1), generation.streaming(1), torch.inference_mode():
        # Use the real native stream; no answer bank or teacher-forced history.
        start = time.monotonic()
        with (args.out / "packets.jsonl").open("w") as log:
            for c in range(len(pcm)//frame_size):
                due = (c+1)/mimi.frame_rate
                time.sleep(max(0, start+due-time.monotonic()))
                before = time.monotonic()
                codes = mimi.encode(torch.from_numpy(pcm[c*frame_size:(c+1)*frame_size]).cuda()[None, None])
                if c == 0:
                    # Published native inference primes once so the first frame
                    # is consumed rather than replaced by initial stream tokens.
                    generation.step(codes)
                tokens = generation.step(codes)
                text = ""
                audio = None
                if tokens is not None:
                    token = int(tokens[0, 0, 0].item())
                    if token not in (0, 1, 2, 3):
                        text = tokenizer.id_to_piece(token).replace("▁", " ")
                    audio = mimi.decode(tokens[:, 1:]).detach().float().cpu().numpy().reshape(-1)
                torch.cuda.synchronize()
                completed = time.monotonic()-start
                playback = None
                if audio is not None:
                    if not np.isfinite(audio).all():
                        raise RuntimeError("Nonfinite generated speech")
                    offset = max(len(output), math.ceil(completed*mimi.sample_rate))
                    playback = offset/mimi.sample_rate
                    output = np.concatenate([np.pad(output, (0, offset-len(output))), audio])
                r = {"chunk": c, "input_available_s": due, "input_consumed_s": before-start,
                     "packet_available_s": completed, "compute_s": completed-(before-start),
                     "input_backlog_s": before-start-due, "text": text,
                     "samples": 0 if audio is None else len(audio), "playback_start_s": playback}
                records.append(r)
                log.write(json.dumps(r)+"\n")
                log.flush()
    if not len(output):
        raise RuntimeError("No generated speech")
    sf.write(args.out / "model.wav", output, mimi.sample_rate)
    report = {"text": "".join(r["text"] for r in records),
              "compute_p50_s": float(np.median([r["compute_s"] for r in records])),
              "compute_p95_s": float(np.percentile([r["compute_s"] for r in records], 95)),
              "max_input_backlog_s": max(r["input_backlog_s"] for r in records),
              "final_input_backlog_s": records[-1]["input_backlog_s"],
              "semantic_review": "pending", "human_listening": "pending"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.out / "manifest.jsonl").write_text(json.dumps({"scenario_id": args.scenario,
        "dir": ".", "scenario_class": "native-Moshi-paced-replay", "duration_ms": args.seconds*1000})+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
