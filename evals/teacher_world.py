"""Declared counterfactual TRAIN worlds; approved user audio stays immutable."""
import json
from evals.moshi_model_download import sha256


def apply_world(path, metadata, scenario, ingestion_sha256):
    world = json.loads(path.read_text())
    if (world.get("schema") != "dl-teacher-world-v1" or world.get("source_scenario") != scenario
            or world.get("source_ingestion_sha256") != ingestion_sha256
            or world.get("project") != metadata["project"] or world.get("field") != metadata["field"]
            or not isinstance(world.get("phrase"), str) or len(world["phrase"].split()) != 2
            or world["phrase"] == metadata["phrase"]):
        raise RuntimeError("Counterfactual world changes the question or lacks a distinct declared value")
    return {**metadata, "phrase": world["phrase"]}, {"path": str(path), "sha256": sha256(path), "world": world}


def recording_id(record):
    return record.get("variant_id", record["source_row"]["scenario_id"])


def teacher_metadata(record, source_meta, ingestion_sha256):
    if "world_override" not in record:
        return source_meta
    from pathlib import Path
    expected = record["world_override"]
    metadata, receipt = apply_world(Path(expected["path"]), source_meta,
                                    record["source_row"]["scenario_id"], ingestion_sha256)
    if receipt != expected:
        raise RuntimeError("Counterfactual teacher world changed")
    return metadata
