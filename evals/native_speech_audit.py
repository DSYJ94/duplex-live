"""Plain words: check that generated audio actually says its generated text.

ASR agreement is a speech-path diagnostic, not answer correctness or a human
listening verdict. It must not stamp model output as verified training data.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, required=True)
    args = ap.parse_args()
    from data.synthetic.filter import load_asr, judge_clip, clip_passes, PER_CLIP_WER_BAR
    from data.synthetic.timeline import Event

    rows = [json.loads(line) for line in (args.probe / "results.jsonl").read_text().splitlines()]
    if len(rows) != 6:
        raise ValueError(f"Expected all six completed probe cases, got {len(rows)}")
    pipe = load_asr("openai/whisper-large-v3-turbo")
    output = []
    for row in rows:
        event = Event(ch="assistant", start_ms=0, text=row["answer"])
        audit = judge_clip(pipe, args.probe / row["id"] / "model.wav", event, "Ethan")
        record = {"id": row["id"], "generated_text": row["answer"], "speech_audit": audit,
                  "asr_roundtrip_pass": clip_passes(audit, PER_CLIP_WER_BAR),
                  "semantic_verdict": "pending review", "human_listening": "pending"}
        output.append(record)
        print(json.dumps(record), flush=True)
    (args.probe / "speech_audit.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
