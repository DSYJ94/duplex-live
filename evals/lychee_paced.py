"""Paced native HF incremental Lychee token benchmark, without future audio.

This measures the speech LM only. Vocoding remains a separate offline step and
must not be counted as a live end-to-end audio result.
"""
import json
import time


def run_incremental(framework, audio, out, profiler=None, before_chunk=None, on_event=None, session=None, on_start=None):
    import numpy as np
    import torch
    session = session or framework.create_incremental_stream_session(
        start_speak_token_factor=1., start_listen_token_factor=1.,
        bc_speak_token_factor=1., end_speak_token_factor=1., audio_incremental_mode=True)
    rows, events = [], []
    started = time.monotonic()
    if on_start is not None:on_start(started)
    size = 6400
    try:
        with (out / "incremental-events.jsonl").open("w") as log:
            for offset in range(0, len(audio), size):
                end = min(offset+size, len(audio))
                due = end/16000
                time.sleep(max(0., started+due-time.monotonic()))
                before = time.monotonic()-started
                if before_chunk is not None:
                    session = before_chunk(session, {"input_available_s":due,"input_consumed_s":before,
                        "input_start_s":offset/16000}, events)
                for event in session.advance_stream(audio[offset:end], audio_is_incremental=True,
                        emit_generation_complete=False, flush_audio_tail=end == len(audio)):
                    record = {"available_s": time.monotonic()-started, "input_available_s": due,
                              "event": event}
                    if on_event is not None:
                        on_event(record)
                    # Released runtime events contain only JSON primitives.
                    log.write(json.dumps(record)+"\n")
                    if event["type"] == "speaking_token":
                        events.append(record)
                torch.cuda.synchronize()
                elapsed = time.monotonic()-started
                rows.append({"input_available_s": due, "input_start_s": offset/16000,
                    "input_consumed_s": before, "compute_s": elapsed-before,
                    "input_backlog_s": before-due, "input_samples": end-offset})
                log.flush()
                if profiler is not None:
                    profiler.step()
        prefix = session.get_prefix()
    finally:
        session.close()
    trace = {"generated_input_ids": prefix["input_ids"],
             "generated_stoken_ids": prefix["stoken_ids"],
             "generated_control_ids": prefix["control_input_ids"]}
    decoded = framework.decoder(trace["generated_input_ids"], trace["generated_stoken_ids"], trace["generated_control_ids"])
    (out / "incremental-chunks.json").write_text(json.dumps(rows, indent=2)+"\n")
    stats = {"schema": "dl-lychee-paced-token-v1", "input_seconds": len(audio)/16000,
        "chunk_seconds": .4, "future_audio_supplied": False, "native_incremental_audio_cache": True,
        "compute_p50_s": float(np.percentile([r["compute_s"] for r in rows], 50)),
        "compute_p95_s": float(np.percentile([r["compute_s"] for r in rows], 95)),
        "maximum_input_backlog_s": max(r["input_backlog_s"] for r in rows),
        "wall_seconds_excluding_load_and_vocoder": time.monotonic()-started,
        "spoken_token_events": len(events), "response_events": len(decoded),
        "live_audio_tested": False, "vocoder_mode": "after LM benchmark, offline quality only"}
    stats["operator_profile_enabled"] = profiler is not None
    (out / "paced-report.json").write_text(json.dumps(stats, indent=2)+"\n")
    print(json.dumps(stats), flush=True)
    return {"events": decoded, "trace": trace}, stats
