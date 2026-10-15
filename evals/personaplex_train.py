"""Bounded native PersonaPlex joint speech/text/action fine-tuning.

Use approved TRAIN caches, the accepted fixed voice prefix, exact native delayed
targets and causal ARC rows. Teacher-forced validation is only diagnostic. A new
process must reproduce the saved logits before free-running evaluation.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from evals.moshi_model_download import sha256


MODEL_HASHES = {
    "model.safetensors": "db1290db583cdaa6cb4de444ed279e0b586ca2a372b41434b07a7461c8c0e2f4",
    "tokenizer-e351c8d8-checkpoint125.safetensors": "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
    "tokenizer_spm_32k_3.model": "78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "voice", "template", "proof", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--cache", type=Path, action="append", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()
    if not 8 <= args.steps <= 128 or args.steps % 8:
        ap.error("Development run requires 8-128 steps in blocks of eight")
    import torch
    import torch.nn.functional as F
    import sentencepiece
    from safetensors.torch import save_file, load_file
    from moshi.models import loaders
    from moshi.utils.compile import no_compile, no_cuda_graph
    from evals.personaplex_bundle import PersonaBundle
    from evals.personaplex_adapters import checkpoint_language_layers
    from evals.personaplex_history import delayed_history, delayed_targets
    from evals.personaplex_policy import action_targets
    from evals.recovery_data import validate_cache
    from evals.recovery_result_scope import private_result_frames
    from evals.moshi_training_smoke import tensor_hashes
    torch.manual_seed(424242)
    random.seed(424242)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if any(sha256(args.model / name) != digest for name, digest in MODEL_HASHES.items()):
        raise RuntimeError("Pinned accepted native speech assets changed")
    proof = json.loads((args.proof / "report.json").read_text())
    template_report = json.loads((args.template / "report.json").read_text())
    if (not proof["passed"] or not proof["all_delayed_inputs_exact"] or not proof["all_delayed_targets_exact"]
            or not proof["adapter_proof"]["passed"] or max(proof["joint_logit_errors"].values()) > .001
            or proof["template_sha256"] != sha256(args.template / "fixed-reference.pt")
            or proof["voice_sha256"] != sha256(args.voice) or template_report["voice_sha256"] != sha256(args.voice)
            or not json.loads((args.template / "reload-report.json").read_text())["passed"]):
        raise RuntimeError("Native joint target/adapter proof or accepted voice prefix changed")
    # The driver-link helper creates its own directory; do not precreate out.
    if not args.reload:
        args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_LIBCUDA_PATH"] = subprocess.check_output([
        sys.executable, "-m", "evals.cuda_linker_path", "--out",
        str(args.out / ("driver-reload" if args.reload else "driver"))], text=True).strip()
    lm = loaders.get_moshi_lm(args.model / "model.safetensors", device="cuda", dtype=torch.float32).eval().requires_grad_(False)
    bundle = PersonaBundle(lm)
    learned = bundle.learned_parameters()

    def precise(fixed):
        with torch.inference_mode(), no_compile(), no_cuda_graph():
            return {n:t.cpu() for n,t in bundle(**{k:v.cuda() if isinstance(v, torch.Tensor) else v
                for k,v in fixed.items() if k != "expected"}).items()}

    if args.reload:
        report = json.loads((args.out / "report.json").read_text())
        if (report["checkpoint_sha256"] != sha256(args.out / "adapters.safetensors")
                or report["proof_sha256"] != sha256(args.proof / "report.json")
                or report["fixed_sha256"] != sha256(args.out / "fixed-reference.pt")):
            raise RuntimeError("Training/reload artifacts changed")
        bundle.load_learned(load_file(args.out / "adapters.safetensors", device="cuda"))
        fixed = torch.load(args.out / "fixed-reference.pt", map_location="cpu", weights_only=True)
        actual = precise(fixed)
        errors = {n:float((value-fixed["expected"][n]).abs().max()) for n,value in actual.items()}
        result = {"schema": "dl-personaplex-joint-reload-v1", "fresh_process": True,
            "errors": errors, "passed": max(errors.values()) == 0., "checkpoint_sha256": report["checkpoint_sha256"]}
        (args.out / "reload-report.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        if not result["passed"]:
            raise RuntimeError("Fresh native joint checkpoint reload changed outputs")
        return

    caches = [(path, validate_cache(path)) for path in args.cache]
    base = caches[0][1]
    if any(c["model"] != base["model"] or c["frame_rate"] != 12.5 for _,c in caches):
        raise RuntimeError("Native training caches use different codec/reference pins")
    if any(base["model"]["files"][n]["sha256"] != MODEL_HASHES[n] for n in MODEL_HASHES if n != "model.safetensors"):
        raise RuntimeError("Native vocabulary/codec differs from accepted PersonaPlex")
    rows = [{**row, "cache_root":path, "conversion":Path(cache["conversion"])}
        for path,cache in caches for row in cache["examples"]]
    if len({r["scenario_id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate approved scenes across supplied caches")
    # Only the first base cache may carry sealed held-out data.
    if any(r["split"] != "train" for _,c in caches[1:] for r in c["examples"]):
        raise RuntimeError("Additional supervision must be TRAIN-only")
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if len(train) < 16 or len(validation) < 4 or {r["kind"] for r in train} != {"private", "simple", "clarify", "cancel"}:
        raise RuntimeError("Missing approved native training behavior/folds")
    template = torch.load(args.template / "fixed-reference.pt", map_location="cpu", weights_only=True)
    prefix = template_report["prefix_frames"]
    tokenizer = sentencepiece.SentencePieceProcessor(str(args.model / "tokenizer_spm_32k_3.model"))

    def prepare(row):
        sample = torch.load(row["cache_root"] / row["file"], map_location="cpu", weights_only=True)
        inputs, clean = delayed_history(template["inputs"], prefix, sample["codes"], lm.delays)
        targets = delayed_targets(template["inputs"], prefix, clean, lm.delays)
        reference = torch.cat([torch.zeros(1, prefix, 4096), sample["reference"].float()], dim=1)
        sidecar = json.loads((row["conversion"] / row["scenario_id"] / "conversation.json").read_text())
        actions = action_targets(sample["codes"][0, 0].tolist(), sidecar["timeline"])
        phrase = private_result_frames(sidecar, clean[0, 0].tolist(), tokenizer.encode) if row["kind"] in ("private", "clarify") else []
        fixed = {"inputs": inputs.cuda(), "targets": targets.cuda(), "reference": reference.cuda(), "prefix_frames": prefix}
        return fixed, torch.tensor([actions], device="cuda"), phrase

    def loss_metrics(row):
        fixed, actions, phrase = prepare(row)
        output = bundle(**fixed)
        text_target = fixed["targets"][:, 0]
        if (text_target == 4).any() or (text_target < 0).any():
            raise RuntimeError("RAG marker or undefined text target entered native speech loss")
        text_nll = F.cross_entropy(output["text"][:, 0].flatten(0, 1), text_target.flatten(), reduction="none").view_as(text_target)
        weights = torch.where(text_target >= 5, 1., .1)
        weights[:, phrase] *= 2.
        text_loss = (text_nll*weights).sum()/weights.sum()
        audio_losses, excluded_audio = [], []
        for q in range(8):
            target = fixed["targets"][:, 1+q]
            if (target < -1).any() or (target > lm.card).any():
                raise RuntimeError("Unknown native audio supervision token: "+row["scenario_id"])
            # Native interleaver uses -1 for undefined padding; the initial
            # token (card) can condition the model but is not an acoustic class.
            valid = (target >= 0) & (target < lm.card)
            if not valid.any():
                raise RuntimeError("No valid native audio target")
            audio_losses.append(F.cross_entropy(output["audio"][:, q][valid], target[valid]))
            excluded_audio.append(int((~valid).sum()))
        audio_loss = (100*audio_losses[0]+sum(audio_losses[1:]))/107
        policy_nll = F.cross_entropy(output["policy"].flatten(0, 1), actions.flatten(), reduction="none").view_as(actions)
        positive = actions != 0
        policy_loss = .5*policy_nll[~positive].mean()+.5*policy_nll[positive].mean() if positive.any() else policy_nll.mean()
        predicted = output["policy"].argmax(-1)
        metrics = {"scenario":row["scenario_id"], "kind":row["kind"], "text_nll":float(text_loss.detach()),
            "word_nll":float(text_nll[text_target >= 5].mean().detach()), "audio_nll":float(audio_loss.detach()),
            "policy_nll":float(policy_loss.detach()), "action_targets":int(positive.sum()),
            "excluded_undefined_audio_targets":excluded_audio,
            "action_predictions":int((predicted != 0).sum()), "correct_action_frames":int(((predicted == actions)&positive).sum()),
            "phrase_nll":float(text_nll[:, phrase].mean().detach()) if phrase else None}
        return text_loss+audio_loss+.2*policy_loss, metrics

    def evaluate(label):
        with torch.inference_mode(), no_compile(), no_cuda_graph():
            result = [loss_metrics(r)[1] for r in validation]
        (args.out / f"validation-{label}.json").write_text(json.dumps(result, indent=2)+"\n")
        return result

    frozen = tensor_hashes((n,p) for n,p in lm.named_parameters() if not p.requires_grad)
    initial = tensor_hashes(learned.items())
    fixed, _, _ = prepare(next(r for r in train if r["kind"] == "private"))
    # Native proof established the adapter implementation; this run additionally
    # checks its own zero-initialized bundle against the frozen base text/audio.
    with torch.inference_mode(), no_compile(), no_cuda_graph():
        before = bundle(**fixed)
        from evals.personaplex_depth import native_audio_logits
        h, t = lm.forward_codes(fixed["inputs"])
        a = native_audio_logits(lm, fixed["targets"], h[:, prefix:])
        zero_errors = {"text":float((before["text"]-t[:, :, prefix:]).abs().max()), "audio":float((before["audio"]-a).abs().max())}
    del before,h,t,a
    if max(zero_errors.values()) != 0.:
        raise RuntimeError("Initial bundle changed frozen native speech outputs")
    optimizer = torch.optim.AdamW([
        {"params":[p for n,p in learned.items() if n.startswith("language.")], "lr":2e-5},
        {"params":[p for n,p in learned.items() if not n.startswith("language.")], "lr":1e-3}], weight_decay=0.)
    report = {"schema":"dl-personaplex-joint-train-v1", "repo_commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_hashes":MODEL_HASHES, "voice_sha256":sha256(args.voice), "proof_sha256":sha256(args.proof / "report.json"),
        "template_sha256":sha256(args.template / "fixed-reference.pt"), "prefix_frames":prefix,
        "sources":[{"cache":str(p), "sha256":sha256(p / "cache.json")} for p,_ in caches],
        "training_scenarios":[r["scenario_id"] for r in train], "test_fold_used_for_training_or_selection":False,
        "trainable_count":sum(p.numel() for p in learned.values()), "replaced_linears":bundle.replaced,
        "zero_errors":zero_errors, "validation_before":evaluate("before"), "losses":[],
        "policy_classes":["idle", "delegate", "cancel"], "acoustic_parameters_frozen":True,
        "native_routing_qualified":False, "quality_qualified":False}
    pools = {k:[] for k in ("private", "simple", "clarify", "cancel")}
    started = time.monotonic()
    for step in range(args.steps):
        kind = list(pools)[step % len(pools)]
        if not pools[kind]:
            pools[kind] = random.sample([r for r in train if r["kind"] == kind], sum(r["kind"] == kind for r in train))
        row = pools[kind].pop()
        optimizer.zero_grad(set_to_none=True)
        with no_compile(), no_cuda_graph(), checkpoint_language_layers(lm):
            loss, metrics = loss_metrics(row)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite native joint loss")
            loss.backward()
        gradients = [p.grad for p in learned.values() if p.grad is not None]
        if not gradients or any(not torch.isfinite(g).all() for g in gradients) or not any(g.count_nonzero() for g in gradients):
            raise RuntimeError("Missing/nonfinite native joint gradients")
        if any(p.grad is not None for p in lm.parameters() if not p.requires_grad):
            raise RuntimeError("Frozen native weight acquired gradients")
        norm = torch.nn.utils.clip_grad_norm_(list(learned.values()), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite joint gradient norm")
        optimizer.step()
        metrics.update(step=step+1, gradient_norm=float(norm), elapsed_s=time.monotonic()-started)
        report["losses"].append(metrics)
        print(json.dumps(metrics), flush=True)
        if (step+1) % 8 == 0:
            save_file({n:p.detach().cpu().contiguous() for n,p in learned.items()}, args.out / f"step-{step+1:04d}.safetensors")
    if frozen != tensor_hashes((n,p) for n,p in lm.named_parameters() if not p.requires_grad):
        raise RuntimeError("Frozen native speech weights changed")
    report["validation_after"] = evaluate("after")
    report["changed_tensors"] = sum(initial[n] != digest for n,digest in tensor_hashes(learned.items()).items())
    save_file({n:p.detach().cpu().contiguous() for n,p in learned.items()}, args.out / "adapters.safetensors")
    fixed = {k:v.cpu() if isinstance(v, torch.Tensor) else v for k,v in fixed.items()}
    torch.save({**fixed, "expected":precise(fixed)}, args.out / "fixed-reference.pt")
    torch.save({"optimizer":optimizer.state_dict(), "cpu_rng":torch.get_rng_state(),
        "cuda_rng":torch.cuda.get_rng_state(), "python_rng":random.getstate(), "steps":args.steps}, args.out / "optimizer.pt")
    report.update(passed=True, frozen_weights_unchanged=True, optimizer_steps=args.steps,
        checkpoint_sha256=sha256(args.out / "adapters.safetensors"), fixed_sha256=sha256(args.out / "fixed-reference.pt"),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("passed", "optimizer_steps", "checkpoint_sha256", "peak_allocated_bytes")}), flush=True)


if __name__ == "__main__":
    main()
