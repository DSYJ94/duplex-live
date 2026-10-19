"""Revalidate fresh source approvals and every derived training artifact."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def validate_conversion(data):
    from data.synthetic.quality import checked_rows
    receipt = json.loads((data / "ingestion.json").read_text())
    if receipt.get('schema') == 'dl-native-old25-timing-repair-v1':
        from evals.native_old25_timing_repair import validate_conversion as validate_repair
        return validate_repair(data, receipt)
    if receipt.get("schema") == "dl-composed-native-v1":
        from evals.recovery_compose_data import validate_composition
        return validate_composition(data,receipt)
    if receipt.get("schema") == "dl-teacher-native-v1":
        from evals.recovery_teacher_convert import validate_teacher_conversion
        return validate_teacher_conversion(data,receipt)
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
        native_v2 = receipt.get('schema') == 'dl-native-dialogue-source-v2'
        if native_v2:
            from data.synthetic.native_curriculum_v2 import specifications
            specs = json.loads(json.dumps(specifications()))
            if (json.loads((batch / 'specifications.json').read_text()) != specs
                    or receipt.get('scope') != 'complete-train-only'
                    or receipt.get('full_heldout_qualification_ready') is not False
                    or receipt.get('failed_heldout_clips_never_repaired') is not True):
                raise RuntimeError('Native-v2 source approval does not retain its explicit TRAIN-only boundary')
            if split == 'train' and (len(actual) != 24 or {r['scenario_id'] for r in actual} != {
                    s['scenario_id'] for s in specs if s['split'] == 'train'}):
                raise RuntimeError('Native-v2 lacks the complete approved TRAIN inventory')
        if (not native_v2 or split == 'train') and {r.get("kind") for r in actual} != {"private","simple","clarify","cancel"}:
            raise RuntimeError("A verified fold lacks required behavior supervision")
    return receipt


def validate_cache(cache):
    receipt = json.loads((cache / "cache.json").read_text())
    if receipt.get("schema") == "dl-composed-cache-v1":
        from evals.recovery_compose_cache import validate_cache_composition
        return validate_cache_composition(cache,receipt)
    conversion = Path(receipt["conversion"])
    validate_conversion(conversion)
    if sha256(conversion / "ingestion.json") != receipt["ingestion_sha256"]:
        raise RuntimeError("Cached conversion receipt changed")
    for row in receipt["examples"]:
        if sha256(cache / row["file"]) != row["sha256"]:
            raise RuntimeError("Encoded training codes/reference changed")
    return receipt
