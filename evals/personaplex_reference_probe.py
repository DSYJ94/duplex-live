"""Native single-slot zero-parity and causal reference dataflow proof.

Uses identical approved teacher-forced histories, not free-running speech.
The positive control manually sets a projection; it is not a trained model.
"""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from moshi.models import loaders, LMGen
    from moshi.utils.compile import no_compile
    from evals.personaplex_reference import ReferenceBridge
    from evals.recovery_data import validate_cache
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(424242)
    cache = validate_cache(args.cache)
    row = next(r for r in cache["examples"] if r["scenario_id"] == "recovery-validation-000")
    sample = torch.load(args.cache / row["file"], map_location="cpu", weights_only=True)
    arrival = sample["arrivals"][0]
    injection = math.ceil(arrival["ready_at_s"] / .08)
    count = min(sample["codes"].shape[-1], injection + len(arrival["rows"]) + 8)
    if not 2 <= injection < count:
        raise RuntimeError("Missing genuine before/after reference window")
    args.out.mkdir(parents=True, exist_ok=False)
    lm = loaders.get_moshi_lm(args.model / "model.safetensors", device="cuda").eval().requires_grad_(False)
    codes = sample["codes"][..., :count].cuda()
    bridge = ReferenceBridge(arrival["rows"].shape[-1], lm.dim, rank=64,
                             device=lm.device, dtype=next(lm.parameters()).dtype).eval()
    histories = []
    for mode in ("native", "native-repeat", "zero-bridge", "positive-control"):
        bridge.clear()
        if mode == "positive-control":
            # Deliberate, recorded nonzero control, NOT a trained checkpoint.
            with torch.no_grad():
                bridge.down.weight.zero_()
                bridge.up.weight.zero_()
                for i in range(64):
                    bridge.down.weight[i, i] = 1
                    bridge.up.weight[i, i] = .1
        hook = lm.transformer.register_forward_pre_hook(bridge.temporal_input) if "bridge" in mode or mode == "positive-control" else None
        gen = LMGen(lm, device="cuda", use_sampling=False, return_logits=True)
        history = []
        try:
            with gen.streaming(1), torch.inference_mode(), no_compile():
                # Both native graphs and hook graph warm up on the same history.
                for _ in range(8):
                    gen.step(codes[:, 9:, :1], moshi_tokens=codes[:, 1:9, :1], text_token=codes[:, 0, 0])
                gen.reset_streaming()
                for frame in range(count):
                    bridge.clear()
                    offset = frame - injection
                    if mode in ("zero-bridge", "positive-control") and 0 <= offset < len(arrival["rows"]):
                        bridge.set_row(arrival["rows"][offset])
                    result, logits = gen.step(codes[:, 9:, frame:frame+1],
                        moshi_tokens=codes[:, 1:9, frame:frame+1], text_token=codes[:, 0, frame])
                    history.append(None if logits is None else tuple(t.detach().float().cpu() for t in logits))
            histories.append(history)
        finally:
            if hook is not None:
                hook.remove()
        print(json.dumps({"mode": mode, "frames": count}), flush=True)

    def difference(left, right, start=0, end=None):
        values = []
        for a, b in zip(left[start:end], right[start:end]):
            if (a is None) != (b is None):
                raise RuntimeError("Bridge changed native startup delay")
            if a is not None:
                values.extend(float((x-y).abs().max()) for x, y in zip(a, b))
        return max(values, default=0.)

    repeat = difference(histories[0], histories[1])
    zero = difference(histories[0], histories[2])
    prefix = difference(histories[0], histories[3], end=injection)
    after = difference(histories[0], histories[3], start=injection)
    report = {"schema": "dl-personaplex-reference-proof-v1", "native_repeat_error": repeat,
        "zero_projection_error": zero, "causal_prefix_error": prefix,
        "positive_post_reference_difference": after, "injection_frame": injection,
        "ready_at_s": arrival["ready_at_s"], "reference_rows": len(arrival["rows"]),
        "cache_sha256": sha256(args.cache / "cache.json"), "source_codes_sha256": row["sha256"],
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "attention": "deterministic math SDPA, native CUDA graphs; diagnostic not latency",
        "history": "identical approved teacher-forced text and audio; native delay logic",
        "positive_control": "manually assigned rank-64 projection, not trained or qualified",
        "voice_prompt_tested": False, "native_delegation_token_assumed": False,
        "free_running_reference_use_tested": False, "optimizer_steps": 0,
        "passed": repeat == 0 and zero == 0 and prefix == 0 and after > .001}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Native bridge failed parity or causal influence")


if __name__ == "__main__":
    main()
