"""Revalidate fresh source approvals and every derived training artifact."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def validate_conversion(data):
    from data.synthetic.quality import checked_rows
    receipt = json.loads((data / "ingestion.json").read_text())
    batch = Path(receipt["source_batch"])
    if sha256(batch / "quality.jsonl") != receipt["source_quality_sha256"] or sha256(
            batch / "specifications.json") != receipt["sealed_specs_sha256"]:
        raise RuntimeError("Fresh source approval/specification changed")
    rows = receipt["accepted"]
    if len({r["scenario_id"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate converted conversation")
    checked_rows(batch, rows, role="fresh conditioned RAG training")
    for row in rows:
        source = batch / row["dir"]
        checks = {Path(row["path"]):row["wav_sha256"],Path(row["path"]).with_suffix(".json"):row["sidecar_sha256"],
            source / "timeline.actual.json":row["source_timeline_sha256"],
            source / "stem_assistant.wav":row["source_assistant_sha256"],source / "stem_user.wav":row["source_user_sha256"]}
        if any(sha256(path) != digest for path,digest in checks.items()):
            raise RuntimeError(f"Converted/source artifact changed: {row['scenario_id']}")
    for split in ("train","validation","test"):
        actual = [json.loads(line) for line in (data / f"{split}.jsonl").read_text().splitlines()]
        if actual != [r for r in rows if r["split"] == split]:
            raise RuntimeError("Converted fold membership changed")
        if {r.get("kind") for r in actual} != {"private","simple","clarify","cancel"}:
            raise RuntimeError("A verified fold lacks required behavior supervision")
    return receipt


def validate_cache(cache):
    receipt = json.loads((cache / "cache.json").read_text())
    conversion = Path(receipt["conversion"])
    validate_conversion(conversion)
    if sha256(conversion / "ingestion.json") != receipt["ingestion_sha256"]:
        raise RuntimeError("Cached conversion receipt changed")
    for row in receipt["examples"]:
        if sha256(cache / row["file"]) != row["sha256"]:
            raise RuntimeError("Encoded training codes/reference changed")
    return receipt
