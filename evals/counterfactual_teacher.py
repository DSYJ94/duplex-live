"""32 raw TRAIN-only teachers: identical questions, four different private worlds.

This deliberately breaks project-to-answer memorization. No target is approved
until independent measured audio and actual tool receipts pass the usual gates.
"""
import argparse
import asyncio
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone
from evals.moshi_model_download import sha256
from evals.recovery_data import validate_conversion
from evals.gpt_live_replay import run


async def generate(data, out, resume=False):
    conversion = validate_conversion(data)
    rows = [r for r in conversion["accepted"] if r["split"] == "train" and r["kind"] == "private"]
    if len(rows) != 8:
        raise RuntimeError("Counterfactual collection requires the eight approved private TRAIN stimuli")
    rng = random.Random(15102026)
    values = [f"{colour} {shape}" for colour in ("red", "blue", "green", "yellow", "purple", "orange", "silver", "brown")
              for shape in ("circle", "square", "triangle", "star", "diamond", "oval")]
    if resume and (out / "quality.jsonl").exists():
        raise RuntimeError("Cannot resume a stamped corpus")
    out.mkdir(parents=True, exist_ok=resume)
    (out / "worlds").mkdir(exist_ok=resume)
    manifest = {"collection_schema": "dl-counterfactual-teacher-v3", "scope": "counterfactual-private",
        "purpose": "raw TRAIN teachers; not approved", "source_conversion": str(data),
        "source_ingestion_sha256": sha256(data / "ingestion.json"), "source_model": "gpt-live-1",
        "created_date": datetime.now(timezone.utc).date().isoformat(), "provenance_class": "closed-content",
        "fold": "train", "concurrent_sessions": 2, "maximum_sessions": 32, "variants": [], "recordings": []}
    for row in rows:
        metadata = json.loads(Path(row["path"]).with_suffix(".json").read_text())["timeline"]["meta"]
        for index, phrase in enumerate(rng.sample([v for v in values if v != metadata["phrase"]], 4)):
            variant = f"cf-{row['scenario_id']}-{index}"
            world = {"schema": "dl-teacher-world-v1", "source_scenario": row["scenario_id"],
                "source_ingestion_sha256": manifest["source_ingestion_sha256"], "project": metadata["project"],
                "field": metadata["field"], "phrase": phrase}
            path = out / "worlds" / (variant+".json")
            if resume:
                if json.loads(path.read_text()) != world:
                    raise RuntimeError("Declared counterfactual world changed")
            else:
                path.write_text(json.dumps(world, indent=2)+"\n")
            manifest["variants"].append({"variant_id": variant, "source_row": row,
                "world_override": {"path": str(path), "sha256": sha256(path), "world": world}})
    if resume:
        previous = json.loads((out / "teacher.json").read_text())
        for key in ("collection_schema", "scope", "source_conversion", "source_ingestion_sha256", "maximum_sessions", "variants"):
            if previous[key] != manifest[key]:
                raise RuntimeError("Counterfactual generation scope changed")
        manifest = previous
        for record in manifest["recordings"]:
            for name,key in (("environment.json","environment_sha256"),("events.jsonl","events_sha256"),
                             ("model_out.wav","audio_sha256"),("transcripts.json","transcripts_sha256")):
                if sha256(out / record["dir"] / name) != record[key]:
                    raise RuntimeError("Completed counterfactual recording changed")
    (out / "teacher.json").write_text(json.dumps(manifest, indent=2)+"\n")
    semaphore = asyncio.Semaphore(2)

    async def one(variant):
        async with semaphore:
            sid = variant["variant_id"]
            if (out / sid).exists():
                archive = out / "incomplete"
                archive.mkdir(exist_ok=True)
                destination = archive / sid
                index = 0
                while destination.exists():
                    index += 1
                    destination = archive / f"{sid}-{index}"
                shutil.move(str(out / sid), destination)
            args = SimpleNamespace(batch=Path(conversion["source_batch"]), conversion=data,
                scenario=variant["source_row"]["scenario_id"], purpose="teacher", backend="gpt-6-luna",
                out=out / sid, record_world=Path(variant["world_override"]["path"]))
            await run(args)
            record = {**variant, "dir": sid}
            for name,key in (("environment.json","environment_sha256"),("events.jsonl","events_sha256"),
                             ("model_out.wav","audio_sha256"),("transcripts.json","transcripts_sha256")):
                record[key] = sha256(args.out / name)
            (args.out / "model.wav").symlink_to("model_out.wav")
            manifest["recordings"].append(record)
            (out / "teacher.json").write_text(json.dumps(manifest, indent=2)+"\n")
            (out / "manifest.jsonl").write_text("".join(json.dumps({"scenario_id":r["variant_id"],
                "dir":r["dir"],"scenario_class":"raw_counterfactual_teacher"})+"\n" for r in manifest["recordings"]))
            print(json.dumps({"teacher_recorded":sid,"completed":len(manifest["recordings"])}), flush=True)

    async with asyncio.TaskGroup() as group:
        finished = {r["variant_id"] for r in manifest["recordings"]}
        for variant in manifest["variants"]:
            if variant["variant_id"] not in finished:
                group.create_task(one(variant))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    asyncio.run(generate(args.data, args.out, args.resume))


if __name__ == "__main__":
    main()
