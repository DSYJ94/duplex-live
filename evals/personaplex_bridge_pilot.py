"""Eight-step engineering pilot, frozen PersonaPlex and accepted voice prefix.

Capture the actual native delayed inputs and targets, prove whole-sequence parity,
then train only a new reference projection on one approved TRAIN phrase. This is
not a delegation model or evidence of generalization. Never read test labels.
"""
import argparse
import json
import os
import subprocess
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--voice", type=Path, required=True)
    ap.add_argument("--voice-provenance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    import torch.nn.functional as F
    import sentencepiece
    from safetensors.torch import save_file, load_file
    from moshi.models import loaders, LMGen
    from moshi.offline import wrap_with_system_tags
    from moshi.utils.compile import no_compile, no_cuda_graph
    from evals.personaplex_reference import ReferenceBridge
    from evals.recovery_data import validate_cache
    from evals.recovery_result_scope import private_result_frames
    from evals.moshi_training_smoke import tensor_hashes
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    provenance = json.loads(args.voice_provenance.read_text())
    if (provenance.get("schema") != "dl-voice-codec-probe-v1"
            or provenance.get("source_model") != "gpt-live-1"
            or provenance.get("reference_sha256") != sha256(args.voice)):
        raise RuntimeError("Accepted synthetic voice provenance changed")
    torch.manual_seed(424242)
    lm = loaders.get_moshi_lm(args.model / "model.safetensors", device="cuda", dtype=torch.float32).eval().requires_grad_(False)
    bridge = ReferenceBridge(4096, lm.dim, rank=64, device="cuda", dtype=torch.float32)
    tokenizer_path = args.model / "tokenizer_spm_32k_3.model"
    tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

    def forward(fixed, reference=True):
        embedded = lm.embed_codes(fixed["inputs"])
        if reference:
            embedded = embedded + bridge.transform(fixed["reference"])
        return lm.forward_embeddings(embedded)[1]

    if args.reload:
        report = json.loads((args.out / "report.json").read_text())
        if sha256(args.out / "bridge.safetensors") != report["bridge_sha256"]:
            raise RuntimeError("Saved bridge changed")
        bridge.load_state_dict(load_file(args.out / "bridge.safetensors", device="cuda"), strict=True)
        fixed = torch.load(args.out / "fixed-reference.pt", map_location="cuda", weights_only=True)
        with torch.inference_mode(), no_compile(), no_cuda_graph():
            actual = forward(fixed).cpu()
        error = float((actual-fixed["expected_logits"].cpu()).abs().max())
        result = {"fresh_process": True, "max_abs_logit_error": error,
                  "bridge_sha256": report["bridge_sha256"], "passed": error == 0.}
        (args.out / "reload-report.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        if not result["passed"]:
            raise RuntimeError("Fresh bridge reload changed logits")
        return

    cache = validate_cache(args.cache)
    if sha256(tokenizer_path) != cache["model"]["files"][tokenizer_path.name]["sha256"]:
        raise RuntimeError("Cached native text vocabulary is incompatible")
    mimi_name = "tokenizer-e351c8d8-checkpoint125.safetensors"
    if sha256(args.model / mimi_name) != cache["model"]["files"][mimi_name]["sha256"]:
        raise RuntimeError("Cached audio codec is incompatible")
    row = next(r for r in cache["examples"] if r["scenario_id"] == "recovery-train-000" and r["split"] == "train" and r["kind"] == "private")
    sample = torch.load(args.cache / row["file"], map_location="cpu", weights_only=True)
    sidecar = json.loads((Path(cache["conversion"]) / row["scenario_id"] / "conversation.json").read_text())
    phrase_frames = private_result_frames(sidecar, sample["codes"][0, 0].tolist(), tokenizer.encode)
    if not phrase_frames:
        raise RuntimeError("No measured private result targets")
    count = max(phrase_frames)+2
    codes = sample["codes"][..., :count].cuda().clone()
    # MoshiRAG's custom marker is a literal byte token in PersonaPlex. Keep its
    # frame as native padding; future routing uses an explicit separate head.
    removed_markers = int((codes[:, 0] == 4).sum())
    codes[:, 0][codes[:, 0] == 4] = 3
    references = sample["reference"][:, :count].cuda()
    mimi = loaders.get_mimi(args.model / mimi_name, "cuda").eval().requires_grad_(False)
    gen = LMGen(lm, device="cuda", use_sampling=False, audio_silence_frame_cnt=6,
                sample_rate=24000, frame_rate=12.5)
    gen.load_voice_prompt(str(args.voice))
    gen.text_prompt_tokens = tokenizer.encode(wrap_with_system_tags(
        "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."))
    captures, streamed, current = [], [], {"frame": None}
    original = gen.prepare_step_input

    def prepare(*pos, **kw):
        result = original(*pos, **kw)
        if result is not None:
            inputs, provided, targets, _, _ = result
            captures.append({"inputs": inputs.detach().clone(), "targets": targets[:, 0, 0].detach().clone(),
                "frame": current["frame"],
                "reference": torch.zeros(1, 1, 4096, device="cuda") if current["frame"] is None
                    else references[:, current["frame"]:current["frame"]+1]})
        return result

    def capture_logits(module, inputs, output):
        streamed.append(output.detach().cpu().clone())

    gen.prepare_step_input = prepare
    handle = lm.text_linear.register_forward_hook(capture_logits)
    try:
        with gen.streaming(1), mimi.streaming(1), torch.inference_mode(), no_compile(), no_cuda_graph():
            gen.step_system_prompts(mimi)
            prefix_frames = len(captures)
            for frame in range(count):
                current["frame"] = frame
                gen.step(codes[:, 9:, frame:frame+1], moshi_tokens=codes[:, 1:9, frame:frame+1],
                         text_token=codes[:, 0, frame])
    finally:
        handle.remove()
        gen.prepare_step_input = original
    if len(captures) != len(streamed) or len(captures) != prefix_frames+count:
        raise RuntimeError("Native capture changed target/frame coverage")
    # Copies outside inference_mode become ordinary tensors usable by autograd.
    fixed = {"inputs": torch.cat([r["inputs"] for r in captures], dim=-1).clone(),
             "reference": torch.cat([r["reference"] for r in captures], dim=1).clone(),
             "targets": torch.stack([r["targets"] for r in captures], dim=1).clone()}
    mask = torch.tensor([r["frame"] in phrase_frames for r in captures], device="cuda", dtype=torch.bool)
    if int(mask.sum()) != len(phrase_frames) or fixed["reference"][:, :prefix_frames].count_nonzero():
        raise RuntimeError("Prefix leaked a result or changed measured target scope")
    with torch.inference_mode(), no_compile(), no_cuda_graph():
        native = forward(fixed, reference=False)
        full_error = float((native[:, 0].cpu()-torch.cat(streamed, dim=1)).abs().max())
        zero_error = float((native-forward(fixed)).abs().max())
    if full_error > .001 or zero_error != 0.:
        raise RuntimeError(f"Native streaming/full training mismatch: {full_error}, zero {zero_error}")
    args.out.mkdir(parents=True, exist_ok=False)
    print(json.dumps({"streaming_full_max_error": full_error, "zero_bridge_error": zero_error,
                      "prefix_frames": prefix_frames, "train_frames": count, "result_tokens": int(mask.sum())}), flush=True)
    frozen_before = tensor_hashes(lm.named_parameters())
    initial = tensor_hashes(bridge.named_parameters())
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=1e-3, weight_decay=0.)
    losses = []
    for step in range(8):
        optimizer.zero_grad(set_to_none=True)
        with no_compile(), no_cuda_graph():
            logits = forward(fixed)[:, 0, mask]
            loss = F.cross_entropy(logits.reshape(-1, lm.text_card), fixed["targets"][:, mask].reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite bridge loss")
            loss.backward()
        gradients = [p.grad for p in bridge.parameters() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients) or not any(g.count_nonzero() for g in gradients):
            raise RuntimeError("Missing/nonfinite/zero bridge gradients")
        norm = torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.)
        optimizer.step()
        losses.append({"step": step+1, "measured_result_nll": float(loss.detach()), "gradient_norm": float(norm)})
        print(json.dumps(losses[-1]), flush=True)
    frozen_after = tensor_hashes(lm.named_parameters())
    if frozen_before != frozen_after or initial == tensor_hashes(bridge.named_parameters()):
        raise RuntimeError("Frozen model changed or bridge did not train")
    with torch.inference_mode(), no_compile(), no_cuda_graph():
        expected = forward(fixed).cpu()
        without = forward(fixed, reference=False).cpu()
    prefix_error = float((expected[:, :, :prefix_frames]-without[:, :, :prefix_frames]).abs().max())
    if prefix_error != 0.:
        raise RuntimeError("Trained bridge changed the reference-free voice prefix")
    save_file({n: p.detach().cpu().contiguous() for n, p in bridge.state_dict().items()}, args.out / "bridge.safetensors")
    torch.save({**{k:v.cpu() for k,v in fixed.items()}, "expected_logits": expected}, args.out / "fixed-reference.pt")
    torch.save(optimizer.state_dict(), args.out / "optimizer.pt")
    report = {"schema": "dl-personaplex-bridge-pilot-v1", "optimizer_steps": 8,
        "training_scenarios": [row["scenario_id"]], "test_or_validation_trained": False,
        "cache_sha256": sha256(args.cache / "cache.json"), "source_codes_sha256": row["sha256"],
        "voice_sha256": sha256(args.voice), "native_token_4_piece": tokenizer.id_to_piece(4),
        "markers_replaced_with_native_padding": removed_markers, "routing_trained": False,
        "streaming_full_max_error": full_error, "zero_bridge_error": zero_error,
        "frozen_model_unchanged": True, "reference_free_prefix_error": prefix_error,
        "bridge_sha256": sha256(args.out / "bridge.safetensors"), "losses": losses,
        "prefix_frames": prefix_frames, "train_frames": count, "result_tokens": int(mask.sum()),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "free_running_quality_tested": False, "generalization_tested": False,
        "native_voice_quality_after_reference_tested": False, "passed": True}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
