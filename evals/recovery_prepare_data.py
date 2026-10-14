"""Judge every fresh clip and convert only measured, approved conversations.

Whisper timestamps label actual audio, including user word ends for later causal
transcript diagnostics. The sealed script-family folds are retained unchanged.
"""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path

from evals.moshi_model_download import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from data.synthetic.timeline import Timeline
    from data.synthetic.filter import norm_words, wer, rule_checks
    from data.synthetic.quality import stamp_batch, checked_rows
    from data.synthetic.timing_audit import grid_words_by_event

    args.out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (args.batch / "manifest.jsonl").read_text().splitlines()]
    if len(rows) != 40 or len({r["scenario_id"] for r in rows}) != 40:
        raise RuntimeError("Fresh sealed curriculum must contain 40 unique scenes")
    for key in ("script_family",):
        groups = [{r[key] for r in rows if r["split"] == split} for split in ("train", "validation", "test")]
        if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]):
            raise RuntimeError("Script family leaked across sealed folds")
    model = whisper.load_model("turbo", device="cuda")
    judge = {"model": "OpenAI whisper turbo", "whisper": importlib.metadata.version("openai-whisper"),
             "timestamped": importlib.metadata.version("whisper-timestamped")}
    reports, timings, converted = [], [], []
    for row in rows:
        sid = row["scenario_id"]
        source = args.batch / row["dir"]
        timeline = Timeline.load(source / "timeline.actual.json")
        issues, stats = rule_checks(timeline, source)
        clips, alignments, user_words, assistant_measured = [], [], [], []
        for index, event in enumerate(e for e in timeline.events if e.text and e.kind != "marker"):
            path = source / "utterances" / f"{index:02d}_{event.ch}.wav"
            audio, rate = sf.read(path, dtype="float32")
            duration = len(audio) / rate
            if audio.ndim != 1 or not np.isfinite(audio).all() or not .1 <= duration <= 30:
                raise RuntimeError(f"Invalid source waveform: {sid}/{path.name}")
            if abs(duration - (event.actual_end_ms-event.actual_start_ms)/1000) > .003:
                raise RuntimeError("Clip/timeline duration mismatch")
            divisor = math.gcd(rate, 16000)
            pcm = resample_poly(audio, 16000//divisor, rate//divisor).astype(np.float32)
            result = whisper.transcribe(model, pcm, language="en", vad=False,
                best_of=5, beam_size=5, temperature=0., verbose=None)
            error = wer(norm_words(event.text), norm_words(result["text"]))
            words = [w for segment in result["segments"] for w in segment["words"]]
            fail = None
            if not words or error > .08:
                fail = f"strict native alignment WER={error}, words={len(words)}"
            previous = 0.
            for word in words:
                a, b = word["start"], word["end"]
                if not math.isfinite(a+b) or not 0 <= a < b <= duration+.08 or a < previous-.02:
                    fail = "Invalid or overlapping measured word intervals"
                previous = b
            if event.ch == "assistant":
                # Grid labels retain script words only when every position matches.
                if len(words) != len(event.text.split()) or any(
                    norm_words(w["text"]) != norm_words(expected)
                    for w, expected in zip(words, event.text.split())):
                    fail = "Assistant script and measured word positions do not match exactly"
                if fail is None:
                    event.word_onsets_ms = [round(w["start"]*1000) for w in words]
                    assistant_measured.append((event, words))
                    alignments.extend([[w["text"].strip(), [event.actual_start_ms/1000+w["start"],
                        min(event.actual_start_ms/1000+w["end"], event.actual_end_ms/1000)], "SPEAKER_MAIN"] for w in words])
            else:
                user_words.extend({"text":w["text"].strip(), "start_s":event.actual_start_ms/1000+w["start"],
                    "end_s":min(event.actual_start_ms/1000+w["end"],event.actual_end_ms/1000)} for w in words)
            clips.append({"index":index,"ch":event.ch,"kind":event.kind,"gate":"wer", "gate_fail":fail,
                "wer":error,"ref":event.text,"hyp":result["text"],"duration_s":duration,"wav_sha256":sha256(path),
                "words": [{"text":w["text"],"start":w["start"],"end":w["end"]} for w in words]})
            print(json.dumps({"scene":sid,"clip":index,"wer":error,"failure":fail}),flush=True)
        timeline.save(source / "timeline.actual.json")
        asst, grid = grid_words_by_event(timeline)
        offsets = []
        for event, words in assistant_measured:
            placed = [(tick, text) for ei,tick,text in grid if asst[ei] is event]
            if len(placed) != len(words):
                issues.append("Measured assistant words lost by emission grid")
            else:
                offsets.extend(abs(tick*80-(event.actual_start_ms+w["start"]*1000))
                               for (tick,_),w in zip(placed,words))
        timing_ok = bool(offsets) and max(offsets) <= 200 and not issues and not any(c["gate_fail"] for c in clips)
        timings.append({"scenario_id":sid,"timing":"ok" if timing_ok else "fail", "max_offset_ms":max(offsets,default=None),
            "offset_bar_ms":200,"method":"Whisper measured intervals, exact script position match, actual emission-grid replay", "issues":issues})
        reports.append({"scenario_id":sid,"scenario_class":row["scenario_class"],"asr_model":judge,
            "asr_detail":{"per_utterance":clips},"rule_issues":issues,"stats":stats})
        if timing_ok:
            target = args.out / sid
            target.mkdir()
            assistant, rate = sf.read(source / "stem_assistant.wav", dtype="float32")
            user, user_rate = sf.read(source / "stem_user.wav", dtype="float32")
            if rate != 24000 or user_rate != rate or user.shape != assistant.shape:
                raise RuntimeError("Unaligned conversation stems")
            sf.write(target / "conversation.wav", np.stack([assistant,user],axis=1),rate,subtype="FLOAT")
            (target / "conversation.json").write_text(json.dumps({"alignments":alignments,"user_words":user_words,
                "reference_arrivals":timeline.meta["reference_arrivals"],"timeline":timeline.to_dict()},indent=2)+"\n")
            converted.append({**row,"path":str(target / "conversation.wav"),"duration":len(assistant)/rate,
                "wav_sha256":sha256(target / "conversation.wav"),"sidecar_sha256":sha256(target / "conversation.json"),
                "source_timeline_sha256":sha256(source / "timeline.actual.json"),
                "source_assistant_sha256":sha256(source / "stem_assistant.wav"),
                "source_user_sha256":sha256(source / "stem_user.wav")})
        (args.batch / "filter_report.jsonl").write_text("".join(json.dumps(r)+"\n" for r in reports))
        (args.batch / "timing_report.jsonl").write_text("".join(json.dumps(r)+"\n" for r in timings))
    (args.batch / "train_manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    quality = stamp_batch(args.batch,.08,args.batch / "timing_report.jsonl")
    checked_rows(args.batch,converted,role="fresh native RAG conversion")
    receipt = {"source_batch":str(args.batch),"source_quality_sha256":sha256(args.batch / "quality.jsonl"),
        "sealed_specs_sha256":sha256(args.batch / "specifications.json"),"asr_judge":judge,"quality":quality,
        "accepted":converted,"human_listening":"pending","independent_script_folds":True}
    (args.out / "ingestion.json").write_text(json.dumps(receipt,indent=2)+"\n")
    for split, minimum in (("train",16),("validation",4),("test",4)):
        examples = [r for r in converted if r["split"] == split]
        (args.out / f"{split}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in examples))
        if len(examples) < minimum:
            raise RuntimeError(f"Insufficient verified {split} examples: {len(examples)} < {minimum}")
    print(json.dumps({"accepted":len(converted),"total":len(rows),"out":str(args.out)}),flush=True)


if __name__ == "__main__":
    main()
