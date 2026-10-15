"""Fit only foreground actions on native self-history, preserving speech weights."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "data", "model", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--states", type=Path, action="append", required=True)
    ap.add_argument("--steps", type=int, default=512)
    args = ap.parse_args()
    if not 128 <= args.steps <= 2048:
        ap.error("Bounded action-only fit requires 128-2048 steps")
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file, save_file
    from evals.recovery_data import validate_conversion
    from evals.personaplex_train import MODEL_HASHES
    from evals.personaplex_policy import action_targets
    from evals.personaplex_prompts import prompt_receipt
    from evals.personaplex_bundle import PersonaBundle
    from moshi.models import loaders
    from moshi.utils.compile import no_compile, no_cuda_graph
    torch.manual_seed(424242)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    data = validate_conversion(args.data)
    expected = {r["scenario_id"]:r for r in data["accepted"] if r["split"] == "train"}
    reports = sorted(p for root in args.states for p in root.glob("*/report.json"))
    features, labels, receipts, seen, roles = [], [], [], set(), []
    for path in reports:
        report = json.loads(path.read_text())
        row = report["source_row"]
        role = report.get("role_prompt", prompt_receipt("teacher"))
        if role != prompt_receipt(role["name"]):
            raise RuntimeError("Unknown or changed native role instructions")
        roles.append(role)
        sid = row["scenario_id"]
        if (report["schema"] != "dl-personaplex-self-history-v1" or not report["passed"]
                or report["heldout_used"] or report["reference_supplied"] or report["speech_targets_approved"]
                or row != expected.get(sid) or sid in seen or report["model_hashes"] != MODEL_HASHES
                or report["source_conversion"] != str(args.data)
                or report["source_ingestion_sha256"] != sha256(args.data / "ingestion.json")):
            raise RuntimeError("Unapproved/duplicate/non-native self-history states")
        for file, key in (("states.pt", "states_sha256"), ("model.wav", "output_sha256"), ("user_in.wav", "input_sha256")):
            if sha256(path.parent / file) != report[key]:
                raise RuntimeError("Native self-history artifact changed")
        state = torch.load(path.parent / "states.pt", map_location="cpu", weights_only=True)
        hidden, actions = state["hidden"], state["actions"]
        if hidden.shape != (report["frames"], 4096) or actions.shape != (report["frames"],) or not torch.isfinite(hidden).all():
            raise RuntimeError("Invalid captured native state shape/values")
        timeline = json.loads(Path(row["path"]).with_suffix(".json").read_text())["timeline"]
        marker_codes = [3]*report["frames"]
        for event in timeline["events"]:
            if event["ch"] == "assistant" and event["kind"] == "marker":
                marker_codes[round(event["start_ms"]/80)] = 4
        declared = torch.tensor(action_targets(marker_codes, timeline))
        if not torch.equal(actions, declared) or int((actions != 0).sum()) != report["action_targets"]:
            raise RuntimeError("Native state action labels differ from approved clocks")
        # Do not teach contradictory idle labels immediately beside the single
        # approved action frame. No extra positive labels or word times are added.
        keep = torch.ones_like(actions, dtype=torch.bool)
        for index in (actions != 0).nonzero().flatten().tolist():
            keep[max(0,index-2):index+3] = False
            keep[index] = True
        features.append(hidden[keep].float())
        labels.append(actions[keep])
        receipts.append({"path":str(path), "sha256":sha256(path), "states_sha256":report["states_sha256"], "scenario":sid})
        seen.add(sid)
    if len(expected) != 23 or seen != set(expected):
        raise RuntimeError("Native policy fit requires all 23 approved TRAIN scenes exactly once")
    if any(role != roles[0] for role in roles):
        raise RuntimeError("Mixed native role prompts in one policy-state fit")
    parent = json.loads((args.parent / "report.json").read_text())
    reload = json.loads((args.parent / "reload-report.json").read_text())
    if (not parent["language_frozen"] or not parent["passed"] or not reload["passed"]
            or parent["checkpoint_sha256"] != sha256(args.parent / "adapters.safetensors")
            or reload["checkpoint_sha256"] != parent["checkpoint_sha256"]
            or max(parent["reference_free_speech_errors"].values()) != 0.
            or any(sha256(args.model / n) != digest for n,digest in MODEL_HASHES.items())):
        raise RuntimeError("Parent native speech/reference checkpoint is unverified")
    if any(json.loads(Path(r["path"]).read_text())["voice_sha256"] != parent["voice_sha256"] for r in receipts):
        raise RuntimeError("Native policy features used a different voice")
    args.out.mkdir(parents=True, exist_ok=False)
    x, y = torch.cat(features).cuda(), torch.cat(labels).cuda()
    groups = [(y == i) for i in range(3)]
    if not all(g.any() for g in groups):
        raise RuntimeError("Native policy feature set lacks an action class")
    policy = torch.nn.Linear(4096, 3, device="cuda", dtype=torch.float32)
    torch.nn.init.zeros_(policy.weight)
    with torch.no_grad():
        policy.bias.copy_(torch.tensor([2.,0.,0.], device="cuda"))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=.001)
    losses = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = policy(x)
        nll = F.cross_entropy(logits, y, reduction="none")
        loss = sum(nll[g].mean() for g in groups)/3
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite self-history policy loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite self-history policy gradient")
        optimizer.step()
        if (step+1) % 64 == 0:
            predicted = logits.detach().argmax(-1)
            metrics = {"step":step+1, "loss":float(loss.detach()),
                "class_frame_accuracy":[float((predicted[g] == y[g]).float().mean()) for g in groups],
                "action_predictions":int((predicted != 0).sum()), "gradient_norm":float(norm)}
            losses.append(metrics)
            print(json.dumps(metrics), flush=True)
    learned = load_file(args.parent / "adapters.safetensors", device="cpu")
    before = {n:v.clone() for n,v in learned.items() if not n.startswith("policy.")}
    learned.update({"policy."+n:p.detach().cpu().contiguous() for n,p in policy.named_parameters()})
    if any(not torch.equal(v, learned[n]) for n,v in before.items()):
        raise RuntimeError("Action-only fit changed parent speech/reference weights")
    save_file(learned, args.out / "adapters.safetensors")
    torch.save({"optimizer":optimizer.state_dict(), "steps":args.steps}, args.out / "policy-optimizer.pt")
    del x,y,features,labels,before,policy,optimizer
    torch.cuda.empty_cache()
    os.environ["TRITON_LIBCUDA_PATH"] = subprocess.check_output([
        sys.executable, "-m", "evals.cuda_linker_path", "--out", str(args.out / "driver")], text=True).strip()
    lm = loaders.get_moshi_lm(args.model / "model.safetensors", device="cuda", dtype=torch.float32).eval().requires_grad_(False)
    bundle = PersonaBundle(lm)
    bundle.load_learned(learned)
    fixed = torch.load(args.parent / "fixed-reference.pt", map_location="cpu", weights_only=True)
    with torch.inference_mode(), no_compile(), no_cuda_graph():
        output = {n:t.cpu() for n,t in bundle(**{k:v.cuda() if isinstance(v,torch.Tensor) else v
            for k,v in fixed.items() if k != "expected"}).items()}
    errors = {n:float((output[n]-fixed["expected"][n]).abs().max()) for n in ("text", "audio")}
    if max(errors.values()) != 0.:
        raise RuntimeError("Action-only fit changed parent speech outputs")
    torch.save({**fixed, "expected":output}, args.out / "fixed-reference.pt")
    report = {**parent, "schema":"dl-personaplex-joint-train-v1", "parent_checkpoint":{
        "path":str(args.parent), "sha256":parent["checkpoint_sha256"], "report_sha256":sha256(args.parent / "report.json")},
        "policy_refit":True, "policy_optimizer_steps":args.steps, "policy_training":receipts,
        "policy_prompt":roles[0],
        "policy_losses":losses, "policy_label_clock":"approved single markers; ignore two neighboring idle frames; no added positives",
        "policy_training_feature_frame_count":sum(json.loads(Path(r["path"]).read_text())["frames"] for r in receipts),
        "parent_speech_errors":errors, "parent_speech_and_reference_weights_unchanged":True,
        "checkpoint_sha256":sha256(args.out / "adapters.safetensors"), "fixed_sha256":sha256(args.out / "fixed-reference.pt"),
        "repo_commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "passed":True, "quality_qualified":False, "native_routing_qualified":False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed":True, "parent_speech_errors":errors, "checkpoint_sha256":report["checkpoint_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
