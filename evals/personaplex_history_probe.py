"""Compare fast prepared histories with actual LMGen on a different TRAIN scene."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "model", "voice", "template", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    args = ap.parse_args()
    import torch
    import sentencepiece
    from moshi.models import loaders, LMGen
    from moshi.offline import wrap_with_system_tags
    from evals.recovery_data import validate_cache
    from evals.personaplex_history import delayed_history
    cache = validate_cache(args.cache)
    report = json.loads((args.template / "report.json").read_text())
    reload = json.loads((args.template / "reload-report.json").read_text())
    if not reload["passed"] or report["voice_sha256"] != sha256(args.voice):
        raise RuntimeError("Unverified prefix template")
    row = next(r for r in cache["examples"] if r["scenario_id"] == "recovery-train-001" and r["split"] == "train")
    sample = torch.load(args.cache / row["file"], map_location="cpu", weights_only=True)
    template = torch.load(args.template / "fixed-reference.pt", map_location="cpu", weights_only=True)
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_LIBCUDA_PATH"] = subprocess.check_output([
        sys.executable, "-m", "evals.cuda_linker_path", "--out", str(args.out / "driver")], text=True).strip()
    lm = loaders.get_moshi_lm(args.model / "model.safetensors", device="cuda").eval().requires_grad_(False)
    mimi = loaders.get_mimi(args.model / "tokenizer-e351c8d8-checkpoint125.safetensors", "cuda").eval().requires_grad_(False)
    expected, codes = delayed_history(template["inputs"], report["prefix_frames"], sample["codes"], lm.delays)
    gen = LMGen(lm, device="cuda", use_sampling=False, audio_silence_frame_cnt=6, sample_rate=24000, frame_rate=12.5)
    gen.load_voice_prompt(str(args.voice))
    tokenizer = sentencepiece.SentencePieceProcessor(str(args.model / "tokenizer_spm_32k_3.model"))
    gen.text_prompt_tokens = tokenizer.encode(wrap_with_system_tags(
        "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."))
    original, observed = gen.prepare_step_input, []

    def prepare(*pos, **kw):
        result = original(*pos, **kw)
        if result is not None:
            observed.append(result[0].detach().cpu().clone())
        return result

    gen.prepare_step_input = prepare
    codes = codes.cuda()
    with gen.streaming(1), mimi.streaming(1), torch.inference_mode():
        gen.step_system_prompts(mimi)
        actual_prefix = len(observed)
        for frame in range(codes.shape[-1]):
            gen.step(codes[:, 9:, frame:frame+1], moshi_tokens=codes[:, 1:9, frame:frame+1],
                     text_token=codes[:, 0, frame])
    actual = torch.cat(observed, dim=-1)
    passed = actual.shape == expected.shape and torch.equal(actual, expected)
    result = {"schema": "dl-personaplex-history-proof-v1", "passed": passed,
        "source_scenario": row["scenario_id"], "split": "train", "heldout_labels_used": False,
        "template_training_scenarios": report["training_scenarios"], "prefix_frames": actual_prefix,
        "input_shape": list(actual.shape), "all_delayed_inputs_exact": passed,
        "cache_sha256": sha256(args.cache / "cache.json"),
        "template_sha256": sha256(args.template / "fixed-reference.pt"), "voice_sha256": sha256(args.voice),
        "quality_tested": False}
    (args.out / "report.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if not passed:
        torch.save({"actual": actual, "expected": expected}, args.out / "mismatch.pt")
        raise RuntimeError("Fast history differs from native LMGen on the second training scene")


if __name__ == "__main__":
    main()
