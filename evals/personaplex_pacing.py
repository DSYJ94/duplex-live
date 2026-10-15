"""Measure the unchanged native PersonaPlex exporter with causally paced input.

Only the user-audio iterator waits. Prompt conditioning and native sampling are
untouched. Capture every generated PCM packet and its actual availability time.
"""
import json
import time
from contextlib import contextmanager


@contextmanager
def paced_export(offline, out, synchronize, before_frame=None):
    import numpy as np
    import soundfile as sf
    from evals.audio_playout import render_packets

    original_iterate = offline.lm_iterate_audio
    original_decode = offline.decode_tokens_to_pcm
    rows, packets = [], []
    state = {"start": None, "current": None}
    report = {}

    def iterate(pcm, sample_interval_size, **kwargs):
        # The official exporter invokes this alias only for the user stream;
        # LMGen's voice/system prompt uses its own unmodified iterator.
        synchronize()
        state["start"] = time.monotonic()
        for index, chunk in enumerate(original_iterate(pcm, sample_interval_size, **kwargs)):
            due = (index + 1) * sample_interval_size / 24000
            time.sleep(max(0., state["start"] + due - time.monotonic()))
            before = time.monotonic() - state["start"]
            row = {"frame": index, "input_available_s": due,
                   "input_consumed_s": before, "input_backlog_s": before - due,
                   "audio_available_s": None, "samples": 0, "tokens": None}
            state["current"] = row
            if before_frame is not None:
                before_frame(row, rows)
            yield chunk
            synchronize()
            row["compute_s"] = time.monotonic() - state["start"] - before
            rows.append(row)
        state["current"] = None

    def decode(mimi, other_mimi, lm_gen, tokens):
        pcm = original_decode(mimi, other_mimi, lm_gen, tokens)
        synchronize()
        row = state["current"]
        if row is None or row["samples"]:
            raise RuntimeError("Native input/output frame mapping changed")
        if mimi.sample_rate != 24000 or len(pcm) != 1920:
            raise RuntimeError("Unexpected native PersonaPlex audio clock")
        available = time.monotonic() - state["start"]
        row.update(audio_available_s=available, samples=len(pcm),
                   tokens=tokens.detach().cpu().reshape(-1).tolist())
        packets.append((available, pcm.copy()))
        return pcm

    offline.lm_iterate_audio, offline.decode_tokens_to_pcm = iterate, decode
    try:
        yield report
    finally:
        offline.lm_iterate_audio, offline.decode_tokens_to_pcm = original_iterate, original_decode
        if packets:
            buffered, ledger = render_packets(packets, 24000, .16)
            unbuffered, old = render_packets(packets, 24000, 0)
            sf.write(out / "model-buffered.wav", buffered, 24000)
            sf.write(out / "model-paced.wav", unbuffered, 24000)
            sf.write(out / "packet-stream.wav", np.concatenate([p for _, p in packets]), 24000)
            (out / "packets.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (out / "playout.json").write_text(json.dumps({"buffered": ledger, "unbuffered": old}, indent=2) + "\n")
            compute = [r["compute_s"] for r in rows]
            report.update(schema="dl-personaplex-paced-v1", input_frames=len(rows),
                input_seconds=len(rows)*.08, compute_p50_s=float(np.percentile(compute, 50)),
                compute_p95_s=float(np.percentile(compute, 95)),
                maximum_input_backlog_s=max(r["input_backlog_s"] for r in rows),
                buffered_underruns=ledger["underrun_count"],
                buffered_underrun_s=ledger["underrun_samples"]/24000,
                unbuffered_underruns=old["underrun_count"],
                source_pcm_samples=ledger["source_samples"],
                wall_seconds_excluding_load_and_prompts=time.monotonic()-state["start"],
                microphone_acceptance=False, delegation_tested=False,
                packet_stream_timing="quality only; paced WAVs retain actual availability")
            (out / "paced-report.json").write_text(json.dumps(report, indent=2) + "\n")
