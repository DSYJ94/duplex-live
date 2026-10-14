"""Causal, bounded audio playout and transparent packet-preserving diagnostics."""
import math


def render_packets(packets, rate=24000, buffer_s=.16):
    """Retain every PCM sample; delay startup once, report every later underrun."""
    import numpy as np
    if rate <= 0 or not 0 <= buffer_s <= .5 or not packets:
        raise ValueError("Invalid bounded playout configuration")
    chunks, ledger, cursor, previous = [], [], 0, -1.
    for index, (available, pcm) in enumerate(packets):
        pcm = np.asarray(pcm)
        if (not math.isfinite(available) or available < previous or available < 0
                or pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all()):
            raise ValueError("Invalid or out-of-order PCM packet")
        earliest = math.ceil(available*rate)
        offset = max(cursor, earliest + (math.ceil(buffer_s*rate) if index == 0 else 0))
        gap = offset-cursor
        chunks.extend([np.zeros(gap, dtype=pcm.dtype), pcm])
        ledger.append({"packet": index, "available_s": available, "samples": len(pcm),
                       "playback_start_sample": offset, "inserted_gap_samples": gap,
                       "underrun_samples": gap if index else 0})
        cursor, previous = offset+len(pcm), available
    return np.concatenate(chunks), {"sample_rate": rate, "startup_buffer_s": buffer_s,
        "packet_count": len(ledger), "source_samples": sum(r["samples"] for r in ledger),
        "underrun_count": sum(r["underrun_samples"] > 0 for r in ledger),
        "underrun_samples": sum(r["underrun_samples"] for r in ledger), "packets": ledger}


def main():
    import argparse
    import json
    from pathlib import Path
    import numpy as np
    import soundfile as sf
    from evals.moshi_model_download import sha256
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--native-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    source = args.native_run / "model.wav"
    audio, rate = sf.read(source, dtype="float32")
    rows = [json.loads(line) for line in (args.native_run / "packets.jsonl").read_text().splitlines()]
    packets, cursor = [], 0
    for row in rows:
        if not row.get("samples"):
            continue
        offset = round(row["playback_start_s"]*rate)
        end = offset+row["samples"]
        if offset < cursor or end > len(audio) or np.count_nonzero(audio[cursor:offset]):
            raise RuntimeError("Packet ledger does not exactly cover captured PCM and inserted zeros")
        packets.append((row["packet_available_s"], audio[offset:end]))
        cursor = end
    if cursor != len(audio):
        raise RuntimeError("Packet ledger leaves unaccounted audio")
    unbuffered, old = render_packets(packets, rate, 0)
    if not np.array_equal(unbuffered, audio):
        raise RuntimeError("Recorded availability does not reproduce original capture exactly")
    buffered, new = render_packets(packets, rate, .16)
    args.out.mkdir(parents=True, exist_ok=False)
    sf.write(args.out / "buffered.wav", buffered, rate)
    sf.write(args.out / "packet-stream.wav", np.concatenate([pcm for _, pcm in packets]), rate)
    report = {"schema": "dl-audio-playout-audit-v1", "source_wav_sha256": sha256(source),
              "source_packets_sha256": sha256(args.native_run / "packets.jsonl"),
              "source": str(args.native_run), "source_capture_reproduced_exactly": True,
              "unbuffered": old, "buffered": new,
              "packet_stream_timing": "quality diagnostic only; no wall-clock latency claim",
              "quality_verdict": "pending human listening; model PCM unchanged"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"unbuffered_underruns": old["underrun_count"],
                      "buffered_underruns": new["underrun_count"],
                      "source_capture_reproduced_exactly": True}), flush=True)


if __name__ == "__main__":
    main()
