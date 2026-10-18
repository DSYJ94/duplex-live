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
from evals.teacher_world import recording_id, teacher_metadata


def validate_input_audio(recorded,source):
    import numpy as np
    import soundfile as sf
    actual,rate = sf.read(recorded,dtype="float32")
    expected,expected_rate = sf.read(source,dtype="float32")
    if rate != 24000 or expected_rate != rate or actual.ndim != 1 or expected.ndim != 1 or not np.isfinite(actual).all():
        raise RuntimeError("Invalid teacher input audio")
    if len(actual) != len(expected)+15*rate or np.max(np.abs(actual[:len(expected)]-expected)) > 1/32768 or np.count_nonzero(actual[len(expected):]):
        raise RuntimeError("Recorded teacher input differs from approved stimulus plus silent tail")


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
        sid = recording_id(record)
        if sid in ids or row["split"] != "train" or row not in source["accepted"]:
            raise RuntimeError("Teacher has duplicate, unapproved or held-out stimuli")
        ids.add(sid)
        folder = batch / record["dir"]
        for name,key in (("environment.json","environment_sha256"),("events.jsonl","events_sha256"),
                         ("model_out.wav","audio_sha256"),("transcripts.json","transcripts_sha256")):
            if sha256(folder / name) != record[key]:
                raise RuntimeError("Raw teacher artifact changed")
        environment = json.loads((folder / "environment.json").read_text())
        if manifest.get("teacher_protocol") or environment.get("teacher_protocol"):
            from evals.teacher_protocol import validate_protocol
            expected_instructions = validate_protocol(manifest.get("teacher_protocol"), "teacher", row["kind"])
            if (environment.get("teacher_protocol") != manifest["teacher_protocol"]
                    or environment["session_config"]["instructions"] != expected_instructions):
                raise RuntimeError("Teacher client-control protocol receipt changed")
        if environment["purpose"] != "teacher" or environment["training_approved"] is not False or environment["input_sha256"] != row["source_user_sha256"]:
            raise RuntimeError("Teacher purpose/stimulus receipt changed")
        if "world_override" in record:
            metadata = json.loads(Path(row["path"]).with_suffix(".json").read_text())["timeline"]["meta"]
            teacher_metadata(record, metadata, manifest["source_ingestion_sha256"])
            if environment.get("record_world_override") != record["world_override"]:
                raise RuntimeError("Counterfactual recording uses a different world")
        elif environment.get("record_world_override"):
            raise RuntimeError("Undeclared teacher world override")
        validate_input_audio(folder / "user_in.wav",Path(source["source_batch"]) / row["dir"] / "stem_user.wav")
    if manifest.get("collection_schema") == "dl-counterfactual-teacher-v3":
        variants = {v["variant_id"]:v for v in manifest["variants"]}
        if len(variants) != 32 or ids != set(variants) or manifest.get("scope") != "counterfactual-private":
            raise RuntimeError("Incomplete or changed counterfactual teacher scope")
        from collections import Counter
        counts = Counter()
        pairs = set()
        for record in manifest["recordings"]:
            row = record["source_row"]
            variant = variants[recording_id(record)]
            if row["kind"] != "private" or any(record.get(k) != variant[k] for k in ("source_row", "world_override")):
                raise RuntimeError("Counterfactual variant differs from sealed declaration")
            pair = (row["scenario_id"], record["world_override"]["world"]["phrase"])
            if pair in pairs:
                raise RuntimeError("Duplicate counterfactual world")
            pairs.add(pair)
            counts[row["scenario_id"]] += 1
        expected_sources = {r["scenario_id"] for r in source["accepted"] if r["split"] == "train" and r["kind"] == "private"}
        if set(counts) != expected_sources or any(count != 4 for count in counts.values()):
            raise RuntimeError("Counterfactual source pairing changed")
    elif manifest.get('collection_schema') == 'dl-native-dialogue-teacher-v2':
        from data.synthetic.native_curriculum_v2 import specifications
        expected_specs = json.loads(json.dumps(specifications()))
        if json.loads((Path(source['source_batch']) / 'specifications.json').read_text()) != expected_specs:
            raise RuntimeError('Native-v2 teacher source scripts changed')
        expected = {r['scenario_id'] for r in expected_specs if r['split'] == 'train'}
        if (len(expected) != 24 or ids != expected or set(manifest['selected_ids']) != expected
                or manifest.get('scope') != 'native-v2' or manifest.get('teacher_protocol') != 'native-dialogue-v2'):
            raise RuntimeError('Native-v2 teacher scope/protocol changed or incomplete')
    elif manifest.get("collection_schema") == "dl-control-teacher-v2":
        expected = {r["scenario_id"] for r in source["accepted"] if r["split"] == "train" and r["kind"] in ("clarify", "cancel")}
        if len(expected) != 7 or ids != expected or set(manifest["selected_ids"]) != expected or manifest.get("scope") != "control":
            raise RuntimeError("Control teacher scope differs from approved TRAIN scenes")
    elif len(ids) != 12 or manifest.get("scope", "legacy") != "legacy":
        raise RuntimeError("Historical teacher scope changed")
    if len(ids) != manifest["maximum_sessions"]:
        raise RuntimeError("Incomplete bounded teacher collection")
    return manifest


def assess(speech,events,metadata,kind,session):
    """Measured audio words are the labels; never invent missing word onsets."""
    reasons = []
    if speech.get("audit_failure"):
        reasons.append(speech["audit_failure"])
    words = [w for segment in speech["asr_segments"] for w in segment.get("words",[])]
    if not session["session_finalized"] or session["errors"]:
        reasons.append("Session failed or did not finalize")
    if not any(e["type"] == "session.closed" for e in events) or any(e["type"] in ("error","worker_failure") for e in events):
        reasons.append("Actual event log lacks a clean finalized session")
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
    if kind in ("private", "clarify"):
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
        if kind == "clarify":
            user = [e for e in metadata["source_events"] if e["ch"] == "user" and e.get("kind", "utterance") == "utterance"]
            if len(user) < 2:
                raise RuntimeError("Clarification source lacks measured identification")
            identifying_end = user[1]["actual_end_ms"]/1000
            identifying_start = user[1]["actual_start_ms"]/1000
            question_tokens = [token for token, start in flat if start < identifying_start]
            if not {"which", "project"}.issubset(question_tokens):
                reasons.append("No measured project clarification before user identification")
            if any(e["received_at_s"] < identifying_end for e in starts):
                reasons.append("Private lookup dispatched before project identification completed")
            required = [{"liquid"}, {"gas"}]
        else:
            required = [{"light"},{"faster"},{"sound"}]
    elif kind == "cancel":
        results = [e for e in events if e["type"] == "local.worker.tool_result"]
        cancellations = [(e, r) for e in results for r in e["result"]["receipts"] if r.get("tool") == "cancel_lookup"]
        if len(cancellations) != 1 or not cancellations[0][1].get("cancelled"):
            reasons.append("Cancellation lacks exactly one actual successful tool receipt")
        else:
            event, receipt = cancellations[0]
            targets = set(receipt.get("task_ids", []))
            prior = {e["delegation"]["id"] for e in starts if e["received_at_s"] < event["received_at_s"] and e["delegation"]["id"] != event["task_id"]}
            if not prior or targets != prior:
                reasons.append("Cancellation receipt does not cancel all prior related lookups")
            acknowledgements = [start for token,start in flat if token == "cancelled"]
            if not acknowledgements or any(start < event["received_at_s"] for start in acknowledgements):
                reasons.append("Spoken cancellation acknowledgement missing or predates actual receipt")
            arrivals = [{"ready_at_s":event["received_at_s"], "text":event["result"]["content"], "task_id":event["task_id"]}]
        if any(r.get("tool") == "lookup_record" for e in results for r in e["result"]["receipts"]):
            reasons.append("Cancelled lookup result was delivered")
        target = spoken_words(metadata["phrase"])
        if any(tokens[i:i+len(target)] == target for i in range(len(tokens)-len(target)+1)):
            reasons.append("Private result was spoken in cancellation scene")
        required = [{"light"}, {"faster"}, {"sound"}]
    elif kind == "simple":
        if starts or any(e["type"].startswith("local.worker.") for e in events):
            reasons.append("Simple task unnecessarily delegated")
        question = next(e["text"] for e in metadata["source_events"] if e["ch"] == "user")
        if metadata.get('curriculum_version') == 2: required = []
        elif "lightning" in question: required = [{"light"},{"faster"},{"sound"}]
        elif "evaporation" in question: required = [{"liquid"},{"gas"}]
        elif "night" in question: required = [{"earth"},{"rotates","rotating","rotation"},{"sun"}]
        elif "triangle" in question: required = [{"three"}]
        else: raise RuntimeError("No independently specified fact check for this teacher question")
    else:
        raise RuntimeError("Unknown bounded teacher scene kind")
    if metadata.get('curriculum_version') == 2:
        required = metadata.get('required_facts')
        if not required or any(not group or not all(isinstance(x, str) and x for x in group) for group in required):
            raise RuntimeError('Native-v2 lacks declared independent answer facts')
        users = [e for e in metadata['source_events'] if e['ch'] == 'user' and e.get('kind', 'utterance') == 'utterance']
        intervals = []
        if 'quiet_until_user_ordinal' in metadata:
            cue = users[metadata['quiet_until_user_ordinal']]
            intervals.append((0., cue['actual_end_ms'] / 1000))
        if 'quiet_between_user_ordinals' in metadata:
            pause, cue = [users[i] for i in metadata['quiet_between_user_ordinals']]
            intervals.append((pause['actual_end_ms'] / 1000 + .4, cue['actual_end_ms'] / 1000))
            later = {token for token, start in flat if start >= cue['actual_end_ms'] / 1000}
            if later.intersection(metadata.get('no_restart_after_cue', [])):
                reasons.append('Paused task restarts an already completed step after the cue')
        if any(w['start'] < b and w['end'] > a for a, b in intervals for w in words):
            reasons.append('Assistant speech intrudes into an explicitly requested quiet interval')
        if kind == 'cancel':
            from data.synthetic.native_curriculum_v2 import QA
            question = users[-1]['text'].split('Instead, ', 1)[1].casefold()
            matches = [facts for q, _, facts in QA['train'] if q.casefold() == question]
            if len(matches) != 1:
                raise RuntimeError('Cancellation follow-up differs from its independently authored question')
            later = {token for token, start in flat if start >= users[-1]['actual_end_ms'] / 1000}
            if any(not later.intersection(group) for group in matches[0]):
                reasons.append('Cancellation omitted the new answer after the actual user request')
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
    sys.argv = [sys.argv[0],"--out",str(args.out),"--device",args.device,"--trim-trailing-silence","--keep-silent-failures","--live-runs",*[
        str(args.batch / record["dir"]) for record in manifest["recordings"]]]
    audit()
    speech = json.loads((args.out / "report.json").read_text())["results"]
    results = []
    for record,spoken in zip(manifest["recordings"],speech):
        folder = args.batch / record["dir"]
        timeline = json.loads(Path(record["source_row"]["path"]).with_suffix(".json").read_text())["timeline"]
        metadata = {**teacher_metadata(record, timeline["meta"], manifest["source_ingestion_sha256"]),"source_events":timeline["events"]}
        result = assess(spoken,[json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()],
            metadata,record["source_row"]["kind"],json.loads((folder / "report.json").read_text()))
        results.append({"scenario_id":recording_id(record),"raw_record":record,**result})
        print(json.dumps({k:v for k,v in results[-1].items() if k not in ("raw_record","measured_words","reference_arrivals","delegation_events")}),flush=True)
    (args.out / "teacher_checks.json").write_text(json.dumps({"schema":"dl-teacher-speech-facts-v1",
        "raw_batch":str(args.batch),"raw_manifest_sha256":sha256(args.batch / "teacher.json"),
        "speech_report_sha256":sha256(args.out / "report.json"),"results":results,
        "training_approved":False,"native_grid_audit":"still required"},indent=2)+"\n")


if __name__ == "__main__":
    main()
