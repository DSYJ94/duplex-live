"""Bounded GPT-Live teacher recordings on approved TRAIN stimuli only.

Targets are raw and unapproved until speech, facts and measured timings pass.
No validation/test audio may be used to generate training targets.
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from evals.recovery_data import validate_conversion
from evals.moshi_model_download import sha256
from evals.gpt_live_replay import run


async def generate(data,out):
    receipt = validate_conversion(data)
    selected = [r for r in receipt["accepted"] if r["split"] == "train" and r["kind"] == "private"] + [
        r for r in receipt["accepted"] if r["split"] == "train" and r["kind"] == "simple"][:4]
    if len(selected) != 12:
        raise RuntimeError("Bounded teacher pass requires eight private and four simple approved train stimuli")
    out.mkdir(parents=True,exist_ok=False)
    manifest = {"purpose":"raw training teacher recordings, not quality-approved", "source_conversion":str(data),
        "source_ingestion_sha256":sha256(data / "ingestion.json"), "source_model":"gpt-live-1",
        "created_date":"2026-10-14","provenance_class":"closed-content","fold":"train",
        "concurrent_sessions":2,"maximum_sessions":12,"recordings":[]}
    (out / "teacher.json").write_text(json.dumps(manifest,indent=2)+"\n")
    semaphore = asyncio.Semaphore(2)
    async def one(row):
        async with semaphore:
            args = SimpleNamespace(batch=Path(receipt["source_batch"]),conversion=data,scenario=row["scenario_id"],
                purpose="teacher",backend="gpt-6-luna",out=out / row["scenario_id"])
            await run(args)
            manifest["recordings"].append({"source_row":row,"dir":row["scenario_id"],
                "environment_sha256":sha256(args.out / "environment.json"),"events_sha256":sha256(args.out / "events.jsonl"),
                "audio_sha256":sha256(args.out / "model_out.wav"),"transcripts_sha256":sha256(args.out / "transcripts.json")})
            (out / "teacher.json").write_text(json.dumps(manifest,indent=2)+"\n")
            print(json.dumps({"teacher_recorded":row["scenario_id"],"completed":len(manifest["recordings"])}),flush=True)
    async with asyncio.TaskGroup() as group:
        for row in selected:
            group.create_task(one(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    asyncio.run(generate(args.data,args.out))


if __name__ == "__main__":
    main()
