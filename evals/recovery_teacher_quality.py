"""Conservative, independently timed checks on raw TRAIN-only GPT-Live audio.

This report alone does not authorize training: native conversion and its grid
audit must also pass. Raw recordings remain immutable and unapproved.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_data import validate_conversion
from evals.recovery_prepare_data import spoken_words


def validate_teacher(batch):
    manifest = json.loads((batch / "teacher.json").read_text())
    if manifest["fold"] != "train" or manifest["source_model"] != "gpt-live-1":
        raise RuntimeError("Teacher scope/model changed")
    conversion = Path(manifest["source_conversion"])
    source = validate_conversion(conversion)
    if sha256(conversion / "ingestion.json") != manifest["source_ingestion_sha256"]:
        raise RuntimeError("Teacher stimuli approval changed")
    ids = set()
    for record in manifest["recordings"]:
        row = record["source_row"]
        sid = row["scenario_id"]
        if sid in ids or row["split"] != "train" or row not in source["accepted"]:
            raise RuntimeError("Teacher has duplicate, unapproved or held-out stimuli")
        ids.add(sid)
        folder = batch / record["dir"]
        for name,key in (("environment.json","environment_sha256"),("events.jsonl","events_sha256"),
                         ("model_out.wav","audio_sha256"),("transcripts.json","transcripts_sha256")):
            if sha256(folder / name) != record[key]:
                raise RuntimeError("Raw teacher artifact changed")
        environment = json.loads((folder / "environment.json").read_text())
        if environment["purpose"] != "teacher" or environment["training_approved"] is not False or environment["input_sha256"] != row["source_user_sha256"]:
            raise RuntimeError("Teacher purpose/stimulus receipt changed")
    if len(ids) != manifest["maximum_sessions"] or len(ids) != 12:
        raise RuntimeError("Incomplete bounded teacher collection")
    return manifest


def assess(speech,events,metadata,kind,session):
    """Measured audio words are the labels; never invent missing word onsets."""
    reasons = []
    words = [w for segment in speech["asr_segments"] for w in segment.get("words",[])]
    if not session["session_finalized"] or session["errors"]:
        reasons.append("Session failed or did not finalize")
    if not words or speech["wer_against_model_text"] > .08:
        reasons.append("Speech agreement exceeds the unchanged per-clip WER bar")
    if spoken_words(speech["model_text"]) != [token for w in words for token in spoken_words(w["text"])]:
        reasons.append("Assistant transcript and measured spoken word positions do not match exactly")
    previous = 0.
    for word in words:
        a,b = word["start"],word["end"]
        if not math.isfinite(a+b) or not 0 <= a < b <= speech["audio_duration_s"]+.04 or a < previous-.02:
            reasons.append("Invalid or overlapping measured word intervals")
            break
        previous = b
    flat = [(token,w["start"]) for w in words for token in spoken_words(w["text"])]
    tokens = [token for token,_ in flat]
    starts = [e for e in events if e["type"] == "session.delegation.created"]
    arrivals = []
    if kind == "private":
        results = [e for e in events if e["type"] == "local.worker.tool_result"]
        if len(starts) != 1 or len(results) != 1:
            reasons.append("Private task lacks exactly one actual worker result")
        else:
            result = results[0]
            receipts = result["result"]["receipts"]
            if len(receipts) != 1 or not all(receipts[0].get(k) == v for k,v in {
                    "executed":True,"read_only":True,"tool":"lookup_record",
                    "project":metadata["project"],"field":metadata["field"],"value":metadata["phrase"]}.items()):
                reasons.append("Actual record receipt does not match the sealed training world")
            ready = result["received_at_s"]
            target = spoken_words(metadata["phrase"])
            mentions = [i for i in range(len(tokens)-len(target)+1) if tokens[i:i+len(target)] == target]
            if not mentions or any(flat[i][1] < ready for i in mentions):
                reasons.append("Checked private phrase missing or spoken before the actual tool result")
            colours = {"red","blue","green","yellow","purple","orange","silver","gold","white","black","pink","brown"}
            shapes = {"circle","square","triangle","star","diamond","oval","heart","cross"}
            if any(tokens[i] in colours and tokens[i+1] in shapes and tokens[i:i+2] != target for i in range(len(tokens)-1)):
                reasons.append("Spoken answer contains a contradictory private phrase")
            arrivals = [{"ready_at_s":ready,"text":result["result"]["content"],"task_id":result["task_id"]}]
        required = [{"light"},{"faster"},{"sound"}]
    elif kind == "simple":
        if starts or any(e["type"].startswith("local.worker.") for e in events):
            reasons.append("Simple task unnecessarily delegated")
        question = next(e["text"] for e in metadata["source_events"] if e["ch"] == "user")
        if "lightning" in question: required = [{"light"},{"faster"},{"sound"}]
        elif "evaporation" in question: required = [{"liquid"},{"gas"}]
        elif "night" in question: required = [{"earth"},{"rotates","rotating","rotation"},{"sun"}]
        elif "triangle" in question: required = [{"three"}]
        else: raise RuntimeError("No independently specified fact check for this teacher question")
    else:
        raise RuntimeError("This teacher pass permits only private and simple training scenes")
    if any(not set(tokens).intersection(group) for group in required):
        reasons.append("Required independently specified answer facts absent in measured speech")
    return {"speech_and_fact_checks_pass":not reasons,"reasons":reasons,"measured_words":words,
        "reference_arrivals":arrivals,"delegation_events":starts,
        "full_task_qualification":False,"training_approved":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args = parser.parse_args()
    manifest = validate_teacher(args.batch)
    # Existing independent audio auditor preserves the original WAV and only
    # trims trailing quiet for Whisper. It does not grant training approval.
    from evals.moshi_speech_audit import main as audit
    sys.argv = [sys.argv[0],"--out",str(args.out),"--device",args.device,"--trim-trailing-silence","--live-runs",*[
        str(args.batch / record["dir"]) for record in manifest["recordings"]]]
    audit()
    speech = json.loads((args.out / "report.json").read_text())["results"]
    results = []
    for record,spoken in zip(manifest["recordings"],speech):
        folder = args.batch / record["dir"]
        timeline = json.loads(Path(record["source_row"]["path"]).with_suffix(".json").read_text())["timeline"]
        metadata = {**timeline["meta"],"source_events":timeline["events"]}
        result = assess(spoken,[json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()],
            metadata,record["source_row"]["kind"],json.loads((folder / "report.json").read_text()))
        results.append({"scenario_id":record["source_row"]["scenario_id"],"raw_record":record,**result})
        print(json.dumps({k:v for k,v in results[-1].items() if k not in ("raw_record","measured_words","reference_arrivals","delegation_events")}),flush=True)
    (args.out / "teacher_checks.json").write_text(json.dumps({"schema":"dl-teacher-speech-facts-v1",
        "raw_batch":str(args.batch),"raw_manifest_sha256":sha256(args.batch / "teacher.json"),
        "speech_report_sha256":sha256(args.out / "report.json"),"results":results,
        "training_approved":False,"native_grid_audit":"still required"},indent=2)+"\n")


if __name__ == "__main__":
    main()
