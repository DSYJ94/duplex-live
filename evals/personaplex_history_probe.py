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
    ap.add_argument("--joint-logits", action="store_true", help="Also prove native text/audio training targets and logits in float32")
    args = ap.parse_args()
    import torch
    import sentencepiece
    from moshi.models import loaders, LMGen
    from moshi.offline import wrap_with_system_tags
    from evals.recovery_data import validate_cache
    from evals.personaplex_history import delayed_history, delayed_targets
    from moshi.utils.compile import no_compile, no_cuda_graph
    if args.joint_logits:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
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
    lm = loaders.get_moshi_lm(args.model / "model.safetensors", device="cuda",
        dtype=torch.float32 if args.joint_logits else torch.bfloat16).eval().requires_grad_(False)
    mimi = loaders.get_mimi(args.model / "tokenizer-e351c8d8-checkpoint125.safetensors", "cuda").eval().requires_grad_(False)
    expected, codes = delayed_history(template["inputs"], report["prefix_frames"], sample["codes"], lm.delays)
    gen = LMGen(lm, device="cuda", use_sampling=False, return_logits=args.joint_logits,
                audio_silence_frame_cnt=6, sample_rate=24000, frame_rate=12.5)
    gen.load_voice_prompt(str(args.voice))
    tokenizer = sentencepiece.SentencePieceProcessor(str(args.model / "tokenizer_spm_32k_3.model"))
    gen.text_prompt_tokens = tokenizer.encode(wrap_with_system_tags(
        "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."))
    original, observed, targets, logits_history = gen.prepare_step_input, [], [], []

    def prepare(*pos, **kw):
        result = original(*pos, **kw)
        if result is not None:
            observed.append(result[0].detach().cpu().clone())
            targets.append(result[2].detach().cpu().clone())
        return result

    gen.prepare_step_input = prepare
    codes = codes.cuda()
    with gen.streaming(1), mimi.streaming(1), torch.inference_mode(), no_compile(), no_cuda_graph():
        gen.step_system_prompts(mimi)
        actual_prefix = len(observed)
        for frame in range(codes.shape[-1]):
            output = gen.step(codes[:, 9:, frame:frame+1], moshi_tokens=codes[:, 1:9, frame:frame+1],
                     text_token=codes[:, 0, frame])
            if args.joint_logits:
                logits_history.append(tuple(v.detach().cpu() for v in output[1]))
    actual = torch.cat(observed, dim=-1)
    inputs_equal = actual.shape == expected.shape and torch.equal(actual, expected)
    passed = inputs_equal
    expected_targets = delayed_targets(template["inputs"], report["prefix_frames"], codes.cpu(), lm.delays)
    target_equal = torch.equal(torch.cat(targets, dim=-1)[..., actual_prefix:], expected_targets)
    joint_errors = None
    adapter_proof = None
    if args.joint_logits:
        from evals.personaplex_depth import native_audio_logits
        with torch.inference_mode(), no_compile(), no_cuda_graph():
            hidden, text = lm.forward_codes(expected.cuda())
            hidden, text = hidden[:, actual_prefix:], text[:, :, actual_prefix:]
            original_audio = lm.forward_depformer_training(expected_targets.cuda(), hidden)
            audio = native_audio_logits(lm, expected_targets.cuda(), hidden)
        stream_audio = torch.stack([v[1] for v in logits_history], dim=2)
        original_audio_errors = (original_audio.cpu()-stream_audio).abs().amax(dim=(0, 2, 3)).tolist()
        joint_errors = {"text": float((text.cpu()-torch.cat([v[0] for v in logits_history], dim=2)).abs().max()),
            "audio": float((audio.cpu()-stream_audio).abs().max())}
        from evals.personaplex_adapters import install_adapters, adapter_parameters, GatedReferenceBridge, checkpoint_language_layers
        replaced = install_adapters(lm)
        bridge = GatedReferenceBridge(4096, lm.dim, rank=64, device="cuda", dtype=torch.float32)
        reference = torch.cat([torch.zeros(1, actual_prefix, 4096), sample["reference"]], dim=1).cuda()

        def adapted():
            embedded = lm.embed_codes(expected.cuda()) + bridge.transform(reference)
            adapted_hidden, adapted_text = lm.forward_embeddings(embedded)
            adapted_hidden, adapted_text = adapted_hidden[:, actual_prefix:], adapted_text[:, :, actual_prefix:]
            return adapted_text, native_audio_logits(lm, expected_targets.cuda(), adapted_hidden)

        with torch.inference_mode(), no_compile(), no_cuda_graph():
            same_text, same_audio = adapted()
        zero_errors = {"text": float((same_text-text).abs().max()), "audio": float((same_audio-audio).abs().max())}
        with no_compile(), no_cuda_graph(), checkpoint_language_layers(lm):
            adapted_text, adapted_audio = adapted()
            import torch.nn.functional as F
            target_text = expected_targets[:, 0].cuda()
            word_mask = target_text >= 5
            word_loss = F.cross_entropy(adapted_text[:, 0][word_mask], target_text[word_mask])
            audio_loss = F.cross_entropy(adapted_audio[:, 0].reshape(-1, lm.card), expected_targets[:, 1].cuda().reshape(-1))
            (word_loss + .1*audio_loss).backward()
        params = adapter_parameters(lm)
        b_grads = {n: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero())
                   for n,p in params.items() if ".lora_B." in n}
        bridge_grads = {n: bool(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero())
                        for n,p in bridge.named_parameters() if n != "down.weight"}
        adapter_proof = {"zero_errors": zero_errors, "replaced_linears": replaced,
            "adapter_parameters": sum(p.numel() for p in params.values()),
            "all_output_adapter_gradients_nonzero_finite": all(b_grads.values()),
            "output_adapter_gradients": b_grads,
            "bridge_gradients_nonzero_finite": all(bridge_grads.values()),
            "frozen_gradients_absent": all(p.grad is None for p in lm.parameters() if not p.requires_grad),
            "optimizer_steps": 0}
        adapter_proof["passed"] = max(zero_errors.values()) == 0. and all(b_grads.values()) and all(bridge_grads.values()) and adapter_proof["frozen_gradients_absent"]
    passed = passed and target_equal and (joint_errors is None or max(joint_errors.values()) <= .001) and (adapter_proof is None or adapter_proof["passed"])
    result = {"schema": "dl-personaplex-history-proof-v1", "passed": passed,
        "source_scenario": row["scenario_id"], "split": "train", "heldout_labels_used": False,
        "template_training_scenarios": report["training_scenarios"], "prefix_frames": actual_prefix,
        "input_shape": list(actual.shape), "all_delayed_inputs_exact": inputs_equal,
        "all_delayed_targets_exact": target_equal, "joint_logit_errors": joint_errors,
        "adapter_proof": adapter_proof,
        "unmodified_batched_audio_errors_by_codebook": original_audio_errors if args.joint_logits else None,
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
