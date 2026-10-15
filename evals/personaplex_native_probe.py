"""Paced native speech/action replay with real asynchronous workers.

The foreground hears arrived audio and chooses actions with its learned head.
Backend transcript context is a causal measured-word oracle, explicitly not live
ASR. No scheduled dispatch, reference oracle, or supplied assistant answer.
"""
import argparse
import concurrent.futures
from contextlib import nullcontext
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "model", "voice", "data", "checkpoint", "arc-source", "manifest", "arc-tokenizer", "out"):
        ap.add_argument("--"+name, type=Path, required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--construction-rng", choices=("preserve", "legacy"), default="preserve",
                    help="Preserve native sampling RNG across parameter construction; legacy is diagnostic only")
    ap.add_argument("--zero-adapters", action="store_true", help="Unadapted speech comparison; head remains idle")
    ap.add_argument("--dispatch-mode", choices=("native", "scheduled-diagnostic"), default="native")
    ap.add_argument("--result-input", choices=("arc", "spoken-text", "native-text-memory"), default="arc",
                    help="Spoken-text is an explicit hybrid diagnostic, not learned result integration")
    ap.add_argument("--instructions", choices=("teacher", "records-v1"), default="teacher")
    ap.add_argument("--worker-word-frames", type=int, choices=range(3,9), default=4)
    ap.add_argument("--worker-handoff", choices=("immediate", "quiet"), default="immediate")
    args = ap.parse_args()
    if args.worker_handoff == "quiet" and args.result_input != "spoken-text":
        raise ValueError("Quiet worker handoff only applies to explicit spoken-text conditioning")
    import numpy as np
    import soundfile as sf
    import torch
    import sentencepiece
    import moshi
    from moshi import offline
    from safetensors.torch import load_file
    from evals.personaplex_train import MODEL_HASHES
    from evals.personaplex_bundle import PersonaBundle
    from evals.personaplex_arc import load_encoder
    from evals.personaplex_pacing import paced_export
    from evals.recovery_data import validate_conversion
    from evals.recovery_context import causal_dialogue
    from evals.recovery_worker import RecordWorld, run_worker, WorkerSuperseded
    if (Path(moshi.__file__).resolve() != (args.source / "moshi/moshi/__init__.py").resolve()
            or subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
            != "3428dfd95309a7f3c84fd93259ded0f810d1ff91"
            or subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True)):
        raise RuntimeError("Native foreground source changed")
    if any(sha256(args.model / name) != digest for name,digest in MODEL_HASHES.items()):
        raise RuntimeError("Native speech assets changed")
    trained = json.loads((args.checkpoint / "report.json").read_text())
    reload = json.loads((args.checkpoint / "reload-report.json").read_text())
    checkpoint_sha = sha256(args.checkpoint / "adapters.safetensors")
    if (trained["schema"] != "dl-personaplex-joint-train-v1" or not trained["passed"]
            or not trained["frozen_weights_unchanged"] or not reload["passed"] or not reload["fresh_process"]
            or max(reload["errors"].values()) != 0. or trained["checkpoint_sha256"] != checkpoint_sha
            or reload["checkpoint_sha256"] != checkpoint_sha or trained["voice_sha256"] != sha256(args.voice)):
        raise RuntimeError("Native joint checkpoint did not pass exact fresh-process reload")
    control_feedback = trained.get("control_feedback", False)
    if control_feedback and (args.result_input != "arc" or args.instructions != "teacher"):
        raise ValueError("Causal state feedback is only trained for native ARC and the original role")
    data = validate_conversion(args.data)
    row = next(r for r in data["accepted"] if r["scenario_id"] == args.scenario)
    if row["split"] != "validation" or row["scenario_id"] in trained["training_scenarios"]:
        raise RuntimeError("Native development probe requires an approved untrained validation scene")
    if args.dispatch_mode != "native" and (row["kind"] != "private" or args.zero_adapters):
        raise ValueError("Scheduled diagnostic requires a trained private validation probe")
    from evals.personaplex_prompts import prompt_receipt
    role = prompt_receipt(args.instructions)
    if args.instructions != "teacher" and args.result_input == "arc":
        raise ValueError("Changed role prompt is only qualified for the explicit worker-text diagnostic")
    if args.dispatch_mode == "native" and not args.zero_adapters and trained.get("policy_prompt", prompt_receipt("teacher")) != role:
        raise RuntimeError("Native policy was trained under a different role prompt")
    sidecar = json.loads(Path(row["path"]).with_suffix(".json").read_text())
    metadata = sidecar["timeline"]["meta"]
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_LIBCUDA_PATH"] = subprocess.check_output([
        sys.executable, "-m", "evals.cuda_linker_path", "--out", str(args.out / "driver")], text=True).strip()
    encoder = load_encoder(args.arc_source, args.manifest, args.arc_tokenizer, args.out) if args.result_input == "arc" else None
    tokenizer = sentencepiece.SentencePieceProcessor(str(args.model / "tokenizer_spm_32k_3.model"))
    from evals.personaplex_result_text import WorkerTextQueue
    text_queue = WorkerTextQueue(tokenizer.encode, args.worker_word_frames)
    pcm, rate = sf.read(row["path"], dtype="float32", always_2d=True)
    if rate != 24000 or pcm.shape[1] != 2 or not np.isfinite(pcm).all():
        raise RuntimeError("Invalid approved native user audio")
    pcm = np.concatenate([pcm[:, 1], np.zeros(20*rate, dtype=np.float32)])
    sf.write(args.out / "user_in.wav", pcm, rate)
    world = RecordWorld({metadata["project"]:{metadata["field"]:metadata["phrase"]}}) if metadata["phrase"] else RecordWorld({})
    environment = {"repo_commit":subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint_sha256":checkpoint_sha, "voice_sha256":sha256(args.voice), "model_hashes":MODEL_HASHES,
        "scenario":args.scenario, "kind":row["kind"], "input_sha256":sha256(args.out / "user_in.wav"),
        "arc_manifest_sha256":sha256(args.manifest), "native_action_choice":args.dispatch_mode == "native",
        "scheduled_dispatch":args.dispatch_mode != "native", "zero_adapters":args.zero_adapters,
        "result_input":args.result_input, "worker_text_forced":args.result_input == "spoken-text",
        "role_prompt":role, "control_feedback":control_feedback, "neutral_ready":trained.get("neutral_ready", False),
        "inference_text_pace_frames":args.worker_word_frames if args.result_input == "spoken-text" else None,
        "worker_handoff":args.worker_handoff,
        "worker_handoff_rule":"five arrived quiet user/output frames; pause worker text on user speech" if args.worker_handoff == "quiet" else None,
        "oracle_reference_rows":False, "backend_context":"causal measured user words and generated assistant tokens; not live ASR",
        "fresh_holdout":False, "microphone_acceptance":False, "action_threshold":.5,
        "repeat_guard":"one dispatch per non-idle action edge; 1.2 second minimum interval",
        "torch":torch.__version__}
    (args.out / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    events, actions, jobs, pending, holder = [], [], {}, [], {}
    original_loader = offline.loaders.get_moshi_lm
    original_gen = offline.LMGen

    class WorkerTextLMGen(original_gen):
        def step(self, input_tokens=None, moshi_tokens=None, text_token=None, return_embeddings=False):
            if holder.get("force_text", False):
                if text_token is not None:
                    raise RuntimeError("Worker text cannot replace native prefix/teacher text")
                text_token = holder["force_token"]
            return super().step(input_tokens=input_tokens, moshi_tokens=moshi_tokens,
                text_token=text_token, return_embeddings=return_embeddings)

    def load_foreground(*pos, **kw):
        lm = original_loader(*pos, **kw).eval().requires_grad_(False)
        # The vendor seeds before calling this loader. Creating Linear adapters
        # advances CUDA RNG even if their learned effect is zero. Preserve the
        # actual native sample stream used by the unadapted state collector.
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(lm.device)
        context = torch.random.fork_rng(devices=[torch.cuda.current_device()]) if args.construction_rng == "preserve" else nullcontext()
        with context:
            bundle = PersonaBundle(lm, control_feedback=control_feedback, neutral_ready=trained.get("neutral_ready", False))
            if not args.zero_adapters:
                bundle.load_learned(load_file(args.checkpoint / "adapters.safetensors", device="cuda"))
            advanced = {"cpu":not torch.equal(cpu_rng,torch.get_rng_state()),
                "cuda":not torch.equal(gpu_rng,torch.cuda.get_rng_state(lm.device))}
        restored = {"cpu":torch.equal(cpu_rng,torch.get_rng_state()),
            "cuda":torch.equal(gpu_rng,torch.cuda.get_rng_state(lm.device))}
        if args.construction_rng == "preserve" and not all(restored.values()):
            raise RuntimeError("Native adapter construction changed the sampled baseline RNG")
        holder["construction_rng"] = {"mode":args.construction_rng,"constructor_advanced_rng":advanced,
            "native_rng_restored":restored}
        environment["construction_rng"] = holder["construction_rng"]
        (args.out / "environment.json").write_text(json.dumps(environment,indent=2)+"\n")
        bundle.eval().requires_grad_(False)
        reference_bridge = bundle.bridge
        if args.result_input == "native-text-memory":
            from evals.personaplex_text_memory import NativeTextMemory
            reference_bridge = NativeTextMemory(lm.text_emb, tokenizer.encode)
        policy_logits = torch.tensor([[2., 0., 0.]], device="cuda")
        def capture_policy(module, inputs):
            policy_logits.copy_(bundle.policy(inputs[0][:, -1].float()))
        holder.update(bundle=bundle, policy_logits=policy_logits,
            reference_bridge=reference_bridge,
            force_token=torch.zeros(1, dtype=torch.long, device="cuda"), force_text=False,
            bridge_hook=lm.transformer.register_forward_pre_hook(bundle.temporal_input if control_feedback else reference_bridge.temporal_input),
            policy_hook=lm.text_linear.register_forward_pre_hook(capture_policy))
        return lm

    def work(task_id, context, revision, dispatch):
        result = run_worker(context, world, task_id, expected_revision=revision)
        receipt = next((r for r in result["receipts"] if r["tool"] == "lookup_record"), None)
        if receipt:
            delay = max(0.,(metadata.get("planned_lookup_ready_s") or 0)-(metadata.get("lookup_trigger_s") or 0))
            time.sleep(max(0.,dispatch+delay-time.monotonic()))
            world.mark_ready(receipt)
        with torch.inference_mode():
            reference = encoder.encode(result["content"]).detach().float().cpu() if encoder is not None else torch.empty(1, 0, 4096)
        if reference.ndim != 3 or reference.shape[0] != 1 or reference.shape[-1] != 4096 or not torch.isfinite(reference).all():
            raise RuntimeError("Invalid completed worker reference")
        return result, reference[0], receipt, time.monotonic()

    started, edge, last_dispatch = None, 0, -10.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        def update_frame(frame, prior):
            nonlocal started, edge, last_dispatch
            if started is None:
                started = time.monotonic()-frame["input_consumed_s"]
            now = time.monotonic()-started
            bundle = holder["bundle"]
            reference_bridge = holder["reference_bridge"]
            reference_bridge.clear()
            for task_id, item in jobs.items():
                if item["observed"] or not item["future"].done():
                    continue
                item["observed"] = True
                try:
                    result, reference, receipt, ready = item["future"].result()
                except WorkerSuperseded:
                    events.append({"event":"worker_stale", "task_id":task_id, "observed_at_s":now})
                    continue
                except Exception as error:
                    events.append({"event":"worker_failed", "task_id":task_id, "observed_at_s":now,
                        "error_type":type(error).__name__, "error":str(error)})
                    continue
                if receipt and not world.deliverable(receipt):
                    events.append({"event":"worker_stale", "task_id":task_id, "observed_at_s":now})
                    continue
                torch.save(reference, args.out / f"reference-{task_id}.pt")
                if args.result_input == "arc":
                    pending.extend((task_id, r, receipt) for r in reference)
                elif args.result_input == "native-text-memory":
                    reference_bridge.append(task_id, result["content"], receipt)
                else:
                    text_queue.append(task_id, result["content"], receipt)
                events.append({"event":"worker_completed", "task_id":task_id, "ready_at_s":ready-started,
                    "observed_at_s":now, "rows":len(reference), "result":result})
            pending[:] = [p for p in pending if p[2] is None or world.deliverable(p[2])]
            if pending:
                task_id, reference, _ = pending.pop(0)
                reference_bridge.set_row(reference)
                events.append({"event":"reference_applied", "task_id":task_id, "frame":frame["frame"], "at_s":now})
            if args.result_input == "native-text-memory":
                memory = reference_bridge.advance(world.deliverable)
                if memory is not None:
                    task_id, token = memory
                    events.append({"event":"native_text_memory", "task_id":task_id,
                        "frame":frame["frame"], "token":token, "at_s":now})
            frame["reference_l1"] = float(reference_bridge.active.abs().sum())
            holder["force_text"] = False
            if args.result_input == "spoken-text":
                user_quiet = frame["input_rms"] < .0015
                last = prior[-1] if prior else {}
                agent_quiet = (last.get("audio_rms", 1.) < .0015 and last.get("tokens")
                    and last["tokens"][0] in (0,3))
                holder["user_quiet_frames"] = holder.get("user_quiet_frames", 0)+1 if user_quiet else 0
                holder["agent_quiet_frames"] = holder.get("agent_quiet_frames", 0)+1 if agent_quiet else 0
                allowed = args.worker_handoff == "immediate" or (holder["user_quiet_frames"] >= 5
                    and (holder.get("worker_text_active", False) or holder["agent_quiet_frames"] >= 5))
                forced = text_queue.pop(world.deliverable) if allowed else None
                if forced is not None:
                    task_id, token = forced
                    holder["worker_text_active"] = True
                    holder["force_token"].fill_(token)
                    holder["force_text"] = True
                    events.append({"event":"worker_text_forced", "task_id":task_id,
                        "frame":frame["frame"], "token":token, "at_s":now})
                elif holder.get("worker_text_active", False) and text_queue.queue and not user_quiet:
                    holder["force_token"].fill_(3)
                    holder["force_text"] = True
                    events.append({"event":"worker_text_paused_for_user", "frame":frame["frame"], "at_s":now})
                elif not text_queue.queue:
                    holder["worker_text_active"] = False
            if frame["frame"] == 0:
                return  # The prefix is conditioning, never a user action.
            probabilities = holder["policy_logits"].softmax(-1)[0].cpu().tolist()
            action = int(np.argmax(probabilities))
            if probabilities[action] < .5:
                action = 0
            action_record = {"frame":frame["frame"], "at_s":now, "probabilities":probabilities, "action":action,
                "input_available_s":frame["input_available_s"], "dispatched":False}
            actions.append(action_record)
            eligible = action != 0 and action != edge and now-last_dispatch >= 1.2
            edge = action
            if args.dispatch_mode == "scheduled-diagnostic":
                first_user = next(e for e in sidecar["timeline"]["events"] if e["ch"] == "user" and e["kind"] == "utterance")
                eligible = not jobs and frame["input_available_s"] >= first_user["actual_end_ms"]/1000+.16
                if eligible:
                    action = 1
                    action_record["scheduled_diagnostic_override"] = True
            if not eligible:
                return
            active = any(not item["observed"] for item in jobs.values())
            if (action == 1 and active) or (action == 2 and not active) or len(jobs) >= 6:
                action_record["blocked"] = "pending task state or bounded session dispatch cap"
                return
            generated = [{"text":tokenizer.id_to_piece(r["tokens"][0]).replace("▁", " "),
                "packet_available_s":r["audio_available_s"]} for r in prior
                if r["tokens"] and r["tokens"][0] >= 5 and r["audio_available_s"] is not None]
            context = causal_dialogue(sidecar, frame["input_available_s"], generated, frame["input_consumed_s"])
            task_id = f"native-{len(jobs)}"
            revision = world.reserve(task_id, context=context)
            dispatch = time.monotonic()
            jobs[task_id] = {"future":pool.submit(work, task_id, context, revision, dispatch), "observed":False, "action":action}
            last_dispatch = now
            action_record["dispatched"] = True
            events.append({"event":"worker_started", "task_id":task_id, "at_s":now,
                "action":action, "context":context, "input_available_s":frame["input_available_s"]})

        def before_frame(frame, prior):
            update_frame(frame, prior)
            if control_feedback:
                from evals.personaplex_control import live_feedback_state, CONTROL_STATES
                with world.lock:
                    cancelled = {identity for identity,task in world.tasks.items() if task["cancelled"]}
                state = live_feedback_state(jobs, cancelled, events)
                holder["bundle"].set_control(state)
                frame["client_feedback_state"] = CONTROL_STATES[state]

        offline.loaders.get_moshi_lm = load_foreground
        if args.result_input == "spoken-text":
            offline.LMGen = WorkerTextLMGen
        try:
            with paced_export(offline, args.out, torch.cuda.synchronize, before_frame) as timing:
                offline.run_inference(input_wav=str(args.out / "user_in.wav"), output_wav=str(args.out / "model_out.wav"),
                    output_text=str(args.out / "output.json"),
                    text_prompt=role["text"],
                    voice_prompt_path=str(args.voice), tokenizer_path=str(args.model / "tokenizer_spm_32k_3.model"),
                    moshi_weight=str(args.model / "model.safetensors"), mimi_weight=str(args.model / "tokenizer-e351c8d8-checkpoint125.safetensors"),
                    hf_repo="nvidia/personaplex-7b-v1", device="cuda", seed=42424242,
                    temp_audio=.8, temp_text=.7, topk_audio=250, topk_text=25, greedy=False, save_voice_prompt_embeddings=False)
        finally:
            offline.loaders.get_moshi_lm = original_loader
            offline.LMGen = original_gen
            for name in ("bridge_hook", "policy_hook"):
                if name in holder:
                    holder[name].remove()
            (args.out / "events.json").write_text(json.dumps(events, indent=2)+"\n")
            (args.out / "actions.jsonl").write_text("".join(json.dumps(a)+"\n" for a in actions))
    tokens = json.loads((args.out / "output.json").read_text())
    report = {"schema":"dl-personaplex-native-probe-v1", "scenario":args.scenario, "kind":row["kind"],
        "text":"".join(t for t in tokens if t not in ("EPAD", "PAD", "BOS", "EOS")), "timing":timing,
        "checkpoint_sha256":checkpoint_sha, "native_action_choice":args.dispatch_mode == "native",
        "scheduled_dispatch":args.dispatch_mode != "native", "zero_adapters":args.zero_adapters,
        "result_input":args.result_input, "worker_text_forced":args.result_input == "spoken-text",
        "role_prompt":role, "control_feedback":control_feedback, "neutral_ready":trained.get("neutral_ready", False),
        "worker_word_frames":args.worker_word_frames, "worker_handoff":args.worker_handoff,
        "backend_context":"causal measured-word oracle; not live ASR", "worker_events":events,
        "worker_dispatches":len(jobs), "unobserved_workers":sum(not j["observed"] for j in jobs.values()),
        "construction_rng":holder["construction_rng"],
        "task_success":"pending independent speech/receipt grading", "quality_qualified":False, "microphone_acceptance":False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    (args.out / "model.wav").symlink_to("model_out.wav")
    print(json.dumps({k:v for k,v in report.items() if k != "worker_events"}), flush=True)


if __name__ == "__main__":
    main()
