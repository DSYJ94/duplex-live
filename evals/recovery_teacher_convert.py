"""Derive immutable native TRAIN audio from independently checked teachers.

Measured word times and actual worker arrival times are retained. Failed
recordings stay excluded; sealed validation/test data is never changed.
"""
import argparse
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_teacher_quality import validate_teacher,assess
from evals.teacher_world import teacher_metadata


def measured_timeline(sid,words,source_timeline,delegations,arrivals):
    from data.synthetic.timeline import Timeline,Event
    from data.synthetic.timing_audit import grid_words_by_event
    if not words or any(len(w["text"].strip().split()) != 1 for w in words):
        raise ValueError("Every teacher word needs its own measured interval")
    start = round(words[0]["start"]*1000)
    end = round(words[-1]["end"]*1000)
    asst = Event("assistant",start," ".join(w["text"].strip() for w in words),
        actual_start_ms=start,actual_end_ms=end,
        word_onsets_ms=[round(w["start"]*1000)-start for w in words])
    events = [Event(**event) for event in source_timeline["events"] if event["ch"] == "user"]
    events.append(asst)
    if delegations:
        first_end = events[0].actual_end_ms / 1000
        for delegation in delegations:
            received = delegation["received_at_s"]
            if received < first_end:
                raise ValueError("Teacher delegation predates complete sealed private question")
            marker = math.ceil(received/.08)*80
            cancel = (source_timeline["meta"].get("kind") == "cancel" and
                any(a["task_id"] == delegation["delegation"]["id"] for a in arrivals))
            events.append(Event("assistant",marker,kind="marker",
                note="delegate actual teacher cancellation" if cancel else "delegate actual teacher lookup"))
    metadata = {**source_timeline["meta"],"scenario_id":sid,"split":"train","source_model":"gpt-live-1",
        "reference_arrivals":arrivals,"verified":False,"teacher_audio":True,
        "timing_scope":"measured output WAV words and actual client tool-result receipt times"}
    timeline = Timeline(sid,"recovery_teacher",events,metadata)
    assistant,grid = grid_words_by_event(timeline)
    if len(assistant) != 1 or len(grid) != len(words):
        raise ValueError("Measured teacher words lost by actual emission grid")
    offsets = [abs(tick*80-word["start"]*1000) for (_,tick,text),word in zip(grid,words)]
    if max(offsets) > 200:
        raise ValueError("Measured teacher word timing exceeds unchanged 200ms grid bar")
    return timeline,max(offsets)


def validate_teacher_conversion(data,receipt):
    from data.synthetic.quality import checked_rows
    raw = Path(receipt["raw_batch"])
    checks = Path(receipt["teacher_checks"])
    raw_manifest = validate_teacher(raw)
    if raw_manifest.get('collection_schema') == 'dl-native-dialogue-teacher-v2':
        from evals.native_v2_semantic_review import verify
        if not receipt.get('semantic_approval') or len(receipt['accepted']) != 24:
            raise RuntimeError('Native-v2 requires whole-corpus semantic approval and all twenty-four recordings')
        semantic = Path(receipt['semantic_approval'])
        if sha256(semantic) != receipt['semantic_approval_sha256']:
            raise RuntimeError('Native-v2 semantic approval changed')
        verify(raw, checks, semantic)
    if sha256(checks) != receipt["teacher_checks_sha256"] or sha256(raw / "teacher.json") != receipt["raw_manifest_sha256"]:
        raise RuntimeError("Teacher approval evidence changed")
    approval = json.loads(checks.read_text())
    if sha256(checks.parent / "report.json") != approval["speech_report_sha256"]:
        raise RuntimeError("Independent teacher speech report changed")
    spoken = {Path(r["run"]).name:r for r in json.loads((checks.parent / "report.json").read_text())["results"]}
    decisions = {"teacher-"+r["scenario_id"]:r for r in approval["results"]}
    passed = {"teacher-"+r["scenario_id"] for r in approval["results"] if r["speech_and_fact_checks_pass"]}
    if any(r["split"] != "train" or r["scenario_id"] not in passed for r in receipt["accepted"]):
        raise RuntimeError("Unapproved teacher or held-out audio entered conversion")
    for row in receipt["accepted"]:
        decision = decisions[row["scenario_id"]]
        record = decision["raw_record"]
        folder = raw / record["dir"]
        source_timeline = json.loads(Path(record["source_row"]["path"]).with_suffix(".json").read_text())["timeline"]
        raw_manifest = json.loads((raw / "teacher.json").read_text())
        reviewed = assess(spoken[decision["scenario_id"]],[json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()],
            {**teacher_metadata(record, source_timeline["meta"], raw_manifest["source_ingestion_sha256"]),"source_events":source_timeline["events"]},row["kind"],
            json.loads((folder / "report.json").read_text()))
        if not reviewed["speech_and_fact_checks_pass"] or reviewed["measured_words"] != decision["measured_words"]:
            raise RuntimeError("Teacher no longer passes current speech/fact/timing checks")
    checked_rows(data,receipt["accepted"],role="measured TRAIN-only teacher conversion")
    for name,key in (("quality.jsonl","source_quality_sha256"),("specifications.json","sealed_specs_sha256")):
        if sha256(data / name) != receipt[key]: raise RuntimeError("Teacher conversion approval changed")
    for row in receipt["accepted"]:
        for name,key in (("conversation.wav","wav_sha256"),("conversation.json","sidecar_sha256"),
            ("timeline.actual.json","source_timeline_sha256"),("stem_assistant.wav","source_assistant_sha256"),
            ("stem_user.wav","source_user_sha256")):
            if sha256(data / row["dir"] / name) != row[key]: raise RuntimeError("Derived teacher artifact changed")
    expected = receipt["accepted"]
    actual = [json.loads(line) for line in (data / "train.jsonl").read_text().splitlines()]
    if actual != expected or any((data / f"{fold}.jsonl").read_text() for fold in ("validation","test")):
        raise RuntimeError("Teacher-only fold membership changed")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument('--semantic-approval', type=Path)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    from data.synthetic.quality import stamp_batch
    checks = json.loads(args.checks.read_text())
    raw = Path(checks["raw_batch"])
    validated_manifest = validate_teacher(raw)
    native_v2 = validated_manifest.get('collection_schema') == 'dl-native-dialogue-teacher-v2'
    if native_v2:
        from evals.native_v2_semantic_review import verify
        if not args.semantic_approval:
            raise RuntimeError('Native-v2 conversion requires independent complete semantic approval')
        verify(raw, args.checks, args.semantic_approval)
    if sha256(raw / "teacher.json") != checks["raw_manifest_sha256"] or sha256(args.checks.parent / "report.json") != checks["speech_report_sha256"]:
        raise RuntimeError("Teacher judge/source hashes changed")
    args.out.mkdir(parents=True,exist_ok=False)
    speech = {Path(r["run"]).name:r for r in json.loads((args.checks.parent / "report.json").read_text())["results"]}
    raw_manifest = json.loads((raw / "teacher.json").read_text())
    rows,reports,timings,rejected = [],[],[],[]
    for result in checks["results"]:
        if not result["speech_and_fact_checks_pass"]:
            rejected.append({"scenario_id":result["scenario_id"],"reasons":result["reasons"]})
            continue
        record = result["raw_record"]
        original = record["source_row"]
        source_sidecar = json.loads(Path(original["path"]).with_suffix(".json").read_text())
        source_sidecar["timeline"]["meta"] = teacher_metadata(record, source_sidecar["timeline"]["meta"], raw_manifest["source_ingestion_sha256"])
        sid = "teacher-"+result["scenario_id"]
        try:
            timeline,offset = measured_timeline(sid,result["measured_words"],source_sidecar["timeline"],
                result["delegation_events"],result["reference_arrivals"])
        except ValueError as error:
            rejected.append({"scenario_id":result["scenario_id"],"reasons":[str(error)]})
            continue
        folder = args.out / sid
        folder.mkdir()
        assistant,rate = sf.read(raw / record["dir"] / "model_out.wav",dtype="float32")
        user,user_rate = sf.read(raw / record["dir"] / "user_in.wav",dtype="float32")
        if rate != 24000 or user_rate != rate or assistant.ndim != 1 or user.ndim != 1 or not np.isfinite(assistant).all() or not np.isfinite(user).all():
            raise RuntimeError("Invalid teacher audio stems")
        length = math.ceil(max(len(assistant),len(user))/1920)*1920
        assistant = np.pad(assistant,(0,length-len(assistant)))
        user = np.pad(user,(0,length-len(user)))
        sf.write(folder / "stem_assistant.wav",assistant,rate,subtype="FLOAT")
        sf.write(folder / "stem_user.wav",user,rate,subtype="FLOAT")
        sf.write(folder / "conversation.wav",np.stack([assistant,user],axis=1),rate,subtype="FLOAT")
        timeline.save(folder / "timeline.actual.json")
        alignments = [[w["text"].strip(),[w["start"],w["end"]],"SPEAKER_MAIN"] for w in result["measured_words"]]
        (folder / "conversation.json").write_text(json.dumps({"alignments":alignments,
            "user_words":source_sidecar["user_words"],"reference_arrivals":result["reference_arrivals"],
            "timeline":timeline.to_dict()},indent=2)+"\n")
        row = {**original,"scenario_id":sid,"dir":sid,"split":"train","verified":True,
            "engine":"gpt-live-1","scenario_class":"recovery_teacher","path":str(folder / "conversation.wav"),
            "duration":length/rate,"duration_ms":round(length/24)}
        for name,key in (("conversation.wav","wav_sha256"),("conversation.json","sidecar_sha256"),
            ("timeline.actual.json","source_timeline_sha256"),("stem_assistant.wav","source_assistant_sha256"),
            ("stem_user.wav","source_user_sha256")): row[key] = sha256(folder / name)
        rows.append(row)
        reports.append({"scenario_id":sid,"scenario_class":"recovery_teacher","asr_model":"Whisper turbo independent words",
            "asr_detail":{"per_utterance":[{"ch":"assistant","kind":"utterance","gate":"wer","gate_fail":None,
                "wer":speech[result["scenario_id"]]["wer_against_model_text"]}]},"verdict":"pass"})
        timings.append({"scenario_id":sid,"timing":"ok","max_offset_ms":offset,"offset_bar_ms":200,
            "method":"Exact transcript/independent measured words; actual grid including real worker marker"})
    if native_v2 and (len(rows) != 24 or rejected):
        raise RuntimeError('Native-v2 conversion cannot approve only a passing subset')
    for name,items in (("manifest.jsonl",rows),("train_manifest.jsonl",rows),("filter_report.jsonl",reports),("timing_report.jsonl",timings)):
        (args.out / name).write_text("".join(json.dumps(row)+"\n" for row in items))
    (args.out / "specifications.json").write_text(json.dumps({"raw_batch":str(raw),"checks_sha256":sha256(args.checks),
        "accepted_ids":[r["scenario_id"] for r in rows],"fold":"train"},indent=2)+"\n")
    stamp_batch(args.out,.08,args.out / "timing_report.jsonl")
    receipt = {"schema":"dl-teacher-native-v1","raw_batch":str(raw),"teacher_checks":str(args.checks),
        "teacher_checks_sha256":sha256(args.checks),"raw_manifest_sha256":checks["raw_manifest_sha256"],
        "source_batch":str(args.out),"source_quality_sha256":sha256(args.out / "quality.jsonl"),
        "sealed_specs_sha256":sha256(args.out / "specifications.json"),"accepted":rows,"rejected":rejected,
        "native_token_grid_check":"Required additionally during codec encoding","human_listening":"pending"}
    if native_v2:
        receipt.update(semantic_approval=str(args.semantic_approval), semantic_approval_sha256=sha256(args.semantic_approval))
    (args.out / "ingestion.json").write_text(json.dumps(receipt,indent=2)+"\n")
    for fold in ("train","validation","test"):
        (args.out / f"{fold}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows if row["split"] == fold))
    validate_teacher_conversion(args.out,receipt)
    print(json.dumps({"accepted":len(rows),"rejected":rejected}),flush=True)


if __name__ == "__main__":
    main()
