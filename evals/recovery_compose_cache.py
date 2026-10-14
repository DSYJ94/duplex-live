"""Reuse approved base codes and add separately audited native teacher codes."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def validate_cache_composition(cache,receipt):
    from evals.recovery_data import validate_cache,validate_conversion
    conversion = Path(receipt["conversion"])
    data = validate_conversion(conversion)
    if data.get("schema") != "dl-composed-native-v1" or sha256(conversion / "ingestion.json") != receipt["ingestion_sha256"]:
        raise RuntimeError("Composed code source approval changed")
    if len(receipt["sources"]) != 2: raise RuntimeError("Expected base and teacher code sources")
    children = []
    for child in receipt["sources"]:
        source = Path(child["path"])
        if sha256(source / "cache.json") != child["cache_sha256"]: raise RuntimeError("Constituent code receipt changed")
        children.append(validate_cache(source))
    if any(child["model"] != receipt["model"] or child["frame_rate"] != receipt["frame_rate"] for child in children):
        raise RuntimeError("Native codec/ARC pinning differs between code sources")
    if [child["conversion"] for child in children] != [child["path"] for child in data["sources"]]:
        raise RuntimeError("Code constituents do not match approved data constituents")
    expected = [row for child in children for row in child["examples"]]
    if expected != receipt["examples"] or len({r["scenario_id"] for r in expected}) != len(expected):
        raise RuntimeError("Composed code membership changed")
    folds = {r["scenario_id"]:r["split"] for r in data["accepted"]}
    if folds != {r["scenario_id"]:r["split"] for r in expected}:
        raise RuntimeError("Composed codes changed approved fold membership")
    for row in expected:
        if sha256(cache / row["file"]) != row["sha256"]: raise RuntimeError("Composed code tensor changed")
    return receipt


def main():
    from evals.recovery_data import validate_cache
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base","teacher","data","out"): parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    base,teacher = validate_cache(args.base),validate_cache(args.teacher)
    args.out.mkdir(parents=True,exist_ok=False)
    receipt = {"schema":"dl-composed-cache-v1","conversion":str(args.data),
        "ingestion_sha256":sha256(args.data / "ingestion.json"),"model":base["model"],"frame_rate":base["frame_rate"],
        "sources":[{"path":str(path),"cache_sha256":sha256(path / "cache.json")} for path in (args.base,args.teacher)],
        "examples":base["examples"]+teacher["examples"],"method":"Reuse hash-verified base codes; add separately verified teacher codes"}
    for source,child in ((args.base,base),(args.teacher,teacher)):
        for row in child["examples"]: (args.out / row["file"]).symlink_to(source / row["file"])
    (args.out / "cache.json").write_text(json.dumps(receipt,indent=2)+"\n")
    validate_cache_composition(args.out,receipt)
    print(json.dumps({"base_reused":len(base["examples"]),"teachers_added":len(teacher["examples"]),"folds_unchanged":True}),flush=True)


if __name__ == "__main__": main()
