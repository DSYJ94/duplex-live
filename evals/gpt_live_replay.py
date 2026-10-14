"""Paced GPT-Live development replay with actual context-based client delegation.

This is a voice reference, not an independent holdout or matched-backend bake-off.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


def audio_sample_count(samples,rate,seconds):
    if rate <= 0 or not 0 < seconds <= 90:
        raise ValueError("Invalid bounded approved input length")
    wanted = round(rate*seconds)
    if wanted < 1 or samples < wanted:
        raise ValueError("Approved input is missing requested samples")
    return wanted


async def run(args):
    import aiohttp
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from websockets.asyncio.client import connect
    from data.synthetic.quality import checked_rows

    scenario = args.scenario
    checked_rows(args.batch, [{"scenario_id": scenario}], role="GPT-Live development replay")
    world,metadata = None,None
    if args.conversion:
        from evals.recovery_data import validate_conversion
        from evals.recovery_worker import RecordWorld
        conversion = validate_conversion(args.conversion)
        row = next(r for r in conversion["accepted"] if r["scenario_id"] == scenario)
        required_split = "train" if args.purpose == "teacher" else "validation"
        if row["split"] != required_split or Path(conversion["source_batch"]) != args.batch:
            raise RuntimeError("Teacher/comparison source does not match its sealed fold")
        metadata = json.loads(Path(row["path"]).with_suffix(".json").read_text())["timeline"]["meta"]
        world = RecordWorld({metadata["project"]:{metadata["field"]:metadata["phrase"]}}) if metadata["phrase"] else RecordWorld({})
    args.out.mkdir(parents=True, exist_ok=False)
    key = os.environ["OPENAI_API_KEY"]
    source = args.batch / scenario / "stem_user.wav"
    pcm, sr = sf.read(source, dtype="float32")
    seconds = len(pcm)/sr if args.conversion else 60
    if pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise RuntimeError("Invalid approved input audio")
    selected_samples = audio_sample_count(len(pcm),sr,seconds)
    divisor = math.gcd(sr, 24000)
    pcm = resample_poly(pcm[:selected_samples], 24000 // divisor, sr // divisor)
    pcm = np.concatenate([pcm, np.zeros(15 * 24000)])
    sf.write(args.out / "user_in.wav", pcm, 24000)
    pcm16 = (np.clip(pcm, -1, 1) * 32767).astype("<i2")
    config = {"model": "gpt-live-1", "delegation": {"type": "client"},
        "instructions": "Be a helpful, concise English voice assistant. Answer straightforward general knowledge yourself. Delegate when you need factual checking or more reasoning. Do not guess or claim a delegated result before it returns. Keep listening while work is pending.",
        "audio": {"format": {"type": "audio/pcm", "rate": 24000}, "output": {"voice": "marin"}}}
    env = {"session_config": config, "backend_model": args.backend,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "fresh_holdout": False, "microphone_acceptance": False,
        "matched_backend_comparison": False, "worker": "actual Responses reasoning on received transcripts",
        "input_seconds": seconds, "added_silence_seconds": 15,
        "scenario":scenario,"actual_record_tools":world is not None,"purpose":args.purpose,
        "provenance_class":"closed-content","source_model":"gpt-live-1","created_date":"2026-10-14",
        "training_approved":False}
    (args.out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    headers = {"Authorization": "Bearer " + key,
        "OpenAI-Safety-Identifier": hashlib.sha256(b"duplex-live-evaluation").hexdigest()}
    transcripts, audio, tasks, errors = [], [], {}, []
    local_starts = []
    usage, started, finalized = None, asyncio.Event(), False
    stream_start = None
    with (args.out / "events.jsonl").open("w") as log:
        def record(event):
            event = {"received_at_s": None if stream_start is None else time.monotonic() - stream_start, **event}
            log.write(json.dumps(event) + "\n")
            log.flush()

        async with aiohttp.ClientSession(headers={"Authorization": "Bearer " + key},
                timeout=aiohttp.ClientTimeout(total=45)) as http:
            async with connect("wss://api.openai.com/v1/live/sessions", additional_headers=headers,
                    open_timeout=20, max_size=2**24) as ws:
                await ws.send(json.dumps({"type": "session.start", "event_id": "replay_start", "session": config}))

                async def worker(delegation_id, context):
                    record({"type": "local.worker.started", "task_id": delegation_id,
                            "context": context, "backend_model": args.backend})
                    payload = {"model": args.backend, "input": json.dumps(context),
                        "instructions": "Help the live voice assistant answer the latest user request using this transcript history. Transcript fragments may be incomplete. Return only relevant facts in at most 100 words. If the request is unclear, identify the missing information. No actions have been executed; do not claim any action completed.",
                        "max_output_tokens": 1000}
                    try:
                        if world is not None:
                            from evals.recovery_worker import run_worker
                            revision = world.reserve(delegation_id,context=context)
                            result = await asyncio.to_thread(run_worker,context,world,delegation_id,
                                model=args.backend,expected_revision=revision)
                            receipt = next((r for r in result["receipts"] if r["tool"] == "lookup_record"),None)
                            delay = max(0.,(metadata.get("planned_lookup_ready_s") or 0)-(metadata.get("lookup_trigger_s") or 0))
                            if receipt:
                                trigger = next(e["received_at_s"] for e in local_starts if e["task_id"] == delegation_id)
                                await asyncio.sleep(max(0.,stream_start+trigger+delay-time.monotonic()))
                                if not world.deliverable(receipt):
                                    record({"type":"local.worker.stale","task_id":delegation_id})
                                    return
                            text = result["content"]
                            data = {"usage": [e["usage"] for e in result["events"]],"id":result["events"][-1]["response_id"]}
                            record({"type":"local.worker.tool_result","task_id":delegation_id,"result":result})
                        else:
                            async with http.post("https://api.openai.com/v1/responses", json=payload) as response:
                                data = await response.json()
                                if response.status != 200 or data.get("status") != "completed":
                                    raise RuntimeError("Backend HTTP/status failure: " + str(response.status))
                            text = " ".join(part["text"] for item in data.get("output", [])
                                if item.get("type") == "message" for part in item.get("content", [])
                                if part.get("type") == "output_text")
                        if not text or len(text.split()) > 150:
                            raise RuntimeError("Missing or excessive backend result")
                        record({"type": "local.worker.completed", "task_id": delegation_id,
                                "content": text, "usage": data.get("usage"), "response_id": data.get("id")})
                        await ws.send(json.dumps({"type": "session.commentary.append",
                            "event_id": "result_" + delegation_id, "delegation_id": delegation_id, "content": text}))
                    except asyncio.CancelledError:
                        record({"type": "local.worker.cancelled", "task_id": delegation_id})
                        raise
                    except Exception as error:
                        errors.append({"type": "worker_failure", "task_id": delegation_id, "error_type": type(error).__name__})
                        record(errors[-1])
                        await ws.send(json.dumps({"type": "session.thinking.append",
                            "delegation_id": delegation_id, "content": "The backend failed. No verified result is available."}))

                async def send():
                    await asyncio.wait_for(started.wait(), timeout=30)
                    for index in range(0, len(pcm16), 1920):
                        await asyncio.sleep(max(0, stream_start + (index + 1920) / 24000 - time.monotonic()))
                        await ws.send(json.dumps({"type": "session.input_audio.append",
                            "audio": base64.b64encode(pcm16[index:index+1920].tobytes()).decode()}))
                    # Leave a small bounded drain window, then finalize for usage.
                    await asyncio.sleep(2)
                    await ws.send(json.dumps({"type": "session.close"}))

                sender = asyncio.create_task(send())
                try:
                    async with asyncio.timeout(seconds+55):
                        async for message in ws:
                            event = json.loads(message)
                            kind = event.get("type")
                            if kind == "session.started":
                                stream_start = time.monotonic()
                                started.set()
                            if kind == "session.output_audio.delta":
                                data = base64.b64decode(event["delta"], validate=True)
                                if len(data) % 2:
                                    raise RuntimeError("Odd PCM16 byte count")
                                audio.append((time.monotonic() - stream_start, data))
                                record({k: v for k, v in event.items() if k != "delta"} | {"audio_bytes": len(data)})
                            else:
                                record(event)
                            if kind in ("session.input_transcript.delta", "session.output_transcript.delta"):
                                transcripts.append({"speaker": "user" if "input" in kind else "assistant",
                                    "text": event["delta"], "start_ms": event.get("start_ms"), "end_ms": event.get("end_ms")})
                            if kind == "session.delegation.created":
                                task_id = event["delegation"]["id"]
                                if task_id in tasks or len(tasks) >= 6:
                                    raise RuntimeError("Duplicate or excessive delegation")
                                local_starts.append({"task_id":task_id,"received_at_s":time.monotonic()-stream_start})
                                tasks[task_id] = asyncio.create_task(worker(task_id, list(transcripts)))
                            if kind == "error":
                                errors.append(event)
                                if not started.is_set():
                                    break
                            if kind == "session.closed":
                                usage, finalized = event.get("usage"), True
                                break
                finally:
                    sender.cancel()
                    for task in tasks.values():
                        task.cancel()
                    await asyncio.gather(sender, *tasks.values(), return_exceptions=True)
    # Align returned audio to local receipt/playback availability, not input time.
    output, cursor = [], 0
    for arrival, data in audio:
        offset = max(cursor, math.ceil(arrival * 24000))
        output.extend([np.zeros(offset - cursor, dtype="<i2"), np.frombuffer(data, dtype="<i2")])
        cursor = offset + len(data) // 2
    sf.write(args.out / "model_out.wav", np.concatenate(output) if output else np.zeros(1), 24000)
    (args.out / "transcripts.json").write_text(json.dumps(transcripts, indent=2) + "\n")
    report = {"session_finalized": finalized, "usage": usage, "audio_packets": len(audio),
        "delegations": len(tasks), "errors": errors, "human_listening": "pending",
        "voice_or_delegation_quality_pass": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not finalized or not audio or errors:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", default="gpt-6-luna")
    parser.add_argument("--scenario",default="dense_dialog-001")
    parser.add_argument("--conversion",type=Path)
    parser.add_argument("--purpose",choices=("comparison","teacher"),default="comparison")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
