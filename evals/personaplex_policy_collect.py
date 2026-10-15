"""Capture causal native states on approved TRAIN user audio, with self-history.

The native model samples all assistant words/audio. Those outputs are evidence,
never training speech labels. Only the already approved action clocks label the
captured states. No worker reference, assistant answer or held-out audio enters.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "model", "voice", "source", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--instructions", choices=("teacher", "records-v1"), default="teacher")
    args = ap.parse_args()
    import torch
    import numpy as np
    import soundfile as sf
    import moshi
    from moshi import offline
    from evals.personaplex_train import MODEL_HASHES
    from evals.recovery_data import validate_conversion
    from evals.personaplex_policy import action_targets
    from evals.personaplex_prompts import prompt_receipt
    role = prompt_receipt(args.instructions)
    if (Path(moshi.__file__).resolve() != (args.source / "moshi/moshi/__init__.py").resolve()
            or subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
            != "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
            or subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True)):
        raise RuntimeError("Native policy-state source changed")
    if any(sha256(args.model / name) != digest for name,digest in MODEL_HASHES.items()):
        raise RuntimeError("Native policy-state model assets changed")
    if sha256(args.voice) != "6d5c88747333eddf3efd65d1a33973b48b0185bd119bcee16ebaae441c6e28e1":
        raise RuntimeError("Accepted voice prefix changed")
    data = validate_conversion(args.data)
    selected = [r for r in data["accepted"] if r["split"] == "train"]
    if len(selected) != 23 or not 0 <= args.index < len(selected):
        raise ValueError("Expected the 23 approved base TRAIN scenes")
    row = selected[args.index]
    sidecar = json.loads(Path(row["path"]).with_suffix(".json").read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_LIBCUDA_PATH"] = subprocess.check_output([
        sys.executable, "-m", "evals.cuda_linker_path", "--out", str(args.out / "driver")], text=True).strip()
    pcm, rate = sf.read(row["path"], dtype="float32", always_2d=True)
    if rate != 24000 or pcm.shape[1] != 2 or not np.isfinite(pcm).all():
        raise RuntimeError("Invalid approved user waveform")
    pcm = np.concatenate([pcm[:, 1], np.zeros(20*rate, dtype=np.float32)])
    sf.write(args.out / "user_in.wav", pcm, rate)
    holder, hidden, frames = {}, [], []
    original_load = offline.loaders.get_moshi_lm
    original_iterate, original_decode = offline.lm_iterate_audio, offline.decode_tokens_to_pcm

    def load(*pos, **kw):
        lm = original_load(*pos, **kw).eval().requires_grad_(False)
        buffer = torch.zeros(1, lm.dim, dtype=torch.bfloat16, device="cuda")
        def capture(module, inputs):
            if inputs[0].dtype != buffer.dtype:
                raise RuntimeError("Native policy-state precision changed")
            buffer.copy_(inputs[0][:, -1])
        holder.update(buffer=buffer, hook=lm.text_linear.register_forward_pre_hook(capture))
        return lm

    def iterate(pcm, sample_interval_size, **kw):
        if sample_interval_size != 1920:
            raise RuntimeError("Native policy frame clock changed")
        for index, chunk in enumerate(original_iterate(pcm, sample_interval_size, **kw)):
            holder["frame"] = index
            yield chunk

    def decode(*pos, **kw):
        output = original_decode(*pos, **kw)
        frame = holder["frame"]
        if frame != len(hidden) or len(output) != 1920:
            raise RuntimeError("Native policy hidden/output clock differs from input clock")
        hidden.append(holder["buffer"].detach().cpu().clone())
        frames.append({"frame":frame, "input_available_s":(frame+1)*.08,
            "tokens":pos[3].detach().cpu().reshape(-1).tolist()})
        return output

    offline.loaders.get_moshi_lm = load
    offline.lm_iterate_audio, offline.decode_tokens_to_pcm = iterate, decode
    try:
        offline.run_inference(input_wav=str(args.out / "user_in.wav"), output_wav=str(args.out / "model.wav"),
            output_text=str(args.out / "output.json"),
            text_prompt=role["text"],
            voice_prompt_path=str(args.voice), tokenizer_path=str(args.model / "tokenizer_spm_32k_3.model"),
            moshi_weight=str(args.model / "model.safetensors"), mimi_weight=str(args.model / "tokenizer-e351c8d8-checkpoint125.safetensors"),
            hf_repo="nvidia/personaplex-7b-v1", device="cuda", seed=42424242,
            temp_audio=.8, temp_text=.7, topk_audio=250, topk_text=25, greedy=False, save_voice_prompt_embeddings=False)
    finally:
        offline.loaders.get_moshi_lm = original_load
        offline.lm_iterate_audio, offline.decode_tokens_to_pcm = original_iterate, original_decode
        if "hook" in holder:
            holder["hook"].remove()
    if len(hidden)*1920 != len(pcm):
        raise RuntimeError("Native self-history state coverage is incomplete")
    marker_codes = [3]*len(hidden)
    for event in sidecar["timeline"]["events"]:
        if event["ch"] == "assistant" and event["kind"] == "marker":
            marker_codes[round(event["start_ms"]/80)] = 4
    targets = action_targets(marker_codes, sidecar["timeline"])
    torch.save({"hidden":torch.cat(hidden), "actions":torch.tensor(targets)}, args.out / "states.pt")
    tokens = json.loads((args.out / "output.json").read_text())
    report = {"schema":"dl-personaplex-self-history-v1", "passed":True, "source_conversion":str(args.data),
        "source_ingestion_sha256":sha256(args.data / "ingestion.json"), "source_row":row,
        "repo_commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_hashes":MODEL_HASHES, "voice_sha256":sha256(args.voice), "input_sha256":sha256(args.out / "user_in.wav"),
        "states_sha256":sha256(args.out / "states.pt"), "output_sha256":sha256(args.out / "model.wav"),
        "frames":len(hidden), "action_targets":sum(v != 0 for v in targets),
        "role_prompt":role,
        "text":"".join(t for t in tokens if t not in ("EPAD", "PAD", "BOS", "EOS")),
        "assistant_history":"native sampled text and audio, not supplied targets", "reference_supplied":False,
        "speech_targets_approved":False, "purpose":"TRAIN-only foreground action states",
        "heldout_used":False, "pacing":"offline causal 80ms model frames, not real-time acceptance"}
    (args.out / "frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in frames))
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
