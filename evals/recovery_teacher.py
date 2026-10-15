"""Bounded GPT-Live teacher recordings on approved TRAIN stimuli only.

Targets are raw and unapproved until speech, facts and measured timings pass.
No validation/test audio may be used to generate training targets.
"""
import argparse
import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from evals.recovery_data import validate_conversion
from evals.moshi_model_download import sha256
from evals.gpt_live_replay import run


async def generate(data,out,resume=False,scope="legacy",protocol=None):
    receipt = validate_conversion(data)
    selected = [r for r in receipt["accepted"] if r["split"] == "train" and r["kind"] == "private"] + [
        r for r in receipt["accepted"] if r["split"] == "train" and r["kind"] == "simple"][:4]
    if len(selected) != 12:
        raise RuntimeError("Bounded teacher pass requires eight private and four simple approved train stimuli")
    if scope == "control":
        selected = [r for r in receipt["accepted"] if r["split"] == "train" and r["kind"] in ("clarify", "cancel")]
        if len(selected) != 7:
            raise RuntimeError("Control teacher pass requires the seven approved clarify/cancel TRAIN stimuli")
    manifest = {"purpose":"raw training teacher recordings, not quality-approved", "source_conversion":str(data),
        "source_ingestion_sha256":sha256(data / "ingestion.json"), "source_model":"gpt-live-1",
        "created_date":datetime.now(timezone.utc).date().isoformat(),"provenance_class":"closed-content","fold":"train",
        "concurrent_sessions":2,"maximum_sessions":len(selected),"recordings":[]}
    if scope == "control":
        manifest.update(collection_schema="dl-control-teacher-v2", scope="control",
                        selected_ids=[r["scenario_id"] for r in selected])
    if protocol:
        from evals.teacher_protocol import validate_protocol
        if scope != "control":
            raise ValueError("Client-control protocol is not a legacy teacher change")
        validate_protocol(protocol, "teacher", selected[0]["kind"])
        manifest["teacher_protocol"] = protocol
    if resume:
        if (out / "quality.jsonl").exists():
            raise RuntimeError("Cannot resume a stamped teacher corpus")
        previous = json.loads((out / "teacher.json").read_text())
        for key in ("source_conversion","source_ingestion_sha256","source_model","fold","maximum_sessions"):
            if previous[key] != manifest[key]:
                raise RuntimeError("Teacher source or generation scope changed")
        if previous.get("scope", "legacy") != scope or previous.get("selected_ids") != manifest.get("selected_ids"):
            raise RuntimeError("Teacher selection changed")
        if previous.get("teacher_protocol") != protocol:
            raise RuntimeError("Teacher client-control protocol changed")
        manifest = previous
        for record in manifest["recordings"]:
            folder = out / record["dir"]
            for name,key in (("environment.json","environment_sha256"),("events.jsonl","events_sha256"),
                             ("model_out.wav","audio_sha256"),("transcripts.json","transcripts_sha256")):
                if sha256(folder / name) != record[key]:
                    raise RuntimeError("Completed unverified teacher artifact changed")
            if not (folder / "model.wav").exists():
                (folder / "model.wav").symlink_to("model_out.wav")
    else:
        out.mkdir(parents=True,exist_ok=False)
    (out / "teacher.json").write_text(json.dumps(manifest,indent=2)+"\n")
    semaphore = asyncio.Semaphore(2)
    async def one(row):
        async with semaphore:
            if (out / row["scenario_id"]).exists():
                archive = out / "incomplete"
                archive.mkdir(exist_ok=True)
                destination = archive / row["scenario_id"]
                index = 0
                while destination.exists():
                    index += 1
                    destination = archive / f"{row['scenario_id']}-{index}"
                shutil.move(str(out / row["scenario_id"]),destination)
            args = SimpleNamespace(batch=Path(receipt["source_batch"]),conversion=data,scenario=row["scenario_id"],
                purpose="teacher",backend="gpt-6-luna",out=out / row["scenario_id"],teacher_protocol=protocol)
            await run(args)
            manifest["recordings"].append({"source_row":row,"dir":row["scenario_id"],
                "environment_sha256":sha256(args.out / "environment.json"),"events_sha256":sha256(args.out / "events.jsonl"),
                "audio_sha256":sha256(args.out / "model_out.wav"),"transcripts_sha256":sha256(args.out / "transcripts.json")})
            (args.out / "model.wav").symlink_to("model_out.wav")
            (out / "manifest.jsonl").write_text("".join(json.dumps({"scenario_id":r["source_row"]["scenario_id"],
                "dir":r["dir"],"scenario_class":"raw_gpt_live_teacher"})+"\n" for r in manifest["recordings"]))
            (out / "teacher.json").write_text(json.dumps(manifest,indent=2)+"\n")
            print(json.dumps({"teacher_recorded":row["scenario_id"],"completed":len(manifest["recordings"])}),flush=True)
    async with asyncio.TaskGroup() as group:
        finished = {r["source_row"]["scenario_id"] for r in manifest["recordings"]}
        for row in selected:
            if row["scenario_id"] not in finished:
                group.create_task(one(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--scope",choices=("legacy", "control"),default="legacy")
    parser.add_argument("--protocol", choices=("client-control-v1",))
    args = parser.parse_args()
    asyncio.run(generate(args.data,args.out,args.resume,args.scope,args.protocol))


if __name__ == "__main__":
    main()
