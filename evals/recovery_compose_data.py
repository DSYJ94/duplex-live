"""Combine independently approved TRAIN teachers with unchanged sealed folds."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def validate_composition(data,receipt):
    from evals.recovery_data import validate_conversion
    if len(receipt["sources"]) != 2:
        raise RuntimeError("Composition requires one sealed base and one train-only teacher source")
    children = []
    for child in receipt["sources"]:
        path = Path(child["path"])
        if sha256(path / "ingestion.json") != child["ingestion_sha256"]:
            raise RuntimeError("Constituent data approval changed")
        children.append(validate_conversion(path))
    base,teacher = children
    if base.get("schema") in ("dl-teacher-native-v1","dl-composed-native-v1") or teacher.get("schema") != "dl-teacher-native-v1":
        raise RuntimeError("Composition source roles changed")
    if len(teacher["accepted"]) < 4 or {r["kind"] for r in teacher["accepted"]} != {"private","simple"}:
        raise RuntimeError("Insufficient independently approved private/simple teachers")
    expected = base["accepted"]+teacher["accepted"]
    if any(row["split"] != "train" for row in teacher["accepted"]) or receipt["accepted"] != expected or len({r["scenario_id"] for r in expected}) != len(expected):
        raise RuntimeError("Teacher data changed the held-out folds or collided with base scenes")
    for split in ("train","validation","test"):
        actual = [json.loads(line) for line in (data / f"{split}.jsonl").read_text().splitlines()]
        if actual != [r for r in expected if r["split"] == split]:
            raise RuntimeError("Composed fold membership changed")
    for row in expected:
        sidecar = Path(row["path"]).with_suffix(".json")
        if sha256(data / row["scenario_id"] / "conversation.json") != row["sidecar_sha256"] or sha256(sidecar) != row["sidecar_sha256"]:
            raise RuntimeError("Composed training sidecar changed")
    return receipt


def main():
    from evals.recovery_data import validate_conversion
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,required=True)
    parser.add_argument("--teacher",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    base = validate_conversion(args.base)
    teacher = validate_conversion(args.teacher)
    # One tiny accepted source can debug serialization, but cannot justify a
    # teacher training comparison. Require both supervised behaviors here.
    if len(teacher["accepted"]) < 4 or {r["kind"] for r in teacher["accepted"]} != {"private","simple"}:
        raise RuntimeError("Insufficient independently approved private/simple teachers")
    args.out.mkdir(parents=True,exist_ok=False)
    receipt = {"schema":"dl-composed-native-v1","sources":[{"path":str(path),
        "ingestion_sha256":sha256(path / "ingestion.json")} for path in (args.base,args.teacher)],
        "accepted":base["accepted"]+teacher["accepted"],"held_out_folds":"unchanged from sealed base",
        "human_listening":"pending"}
    for row in receipt["accepted"]:
        (args.out / row["scenario_id"]).symlink_to(Path(row["path"]).parent,target_is_directory=True)
    for split in ("train","validation","test"):
        (args.out / f"{split}.jsonl").write_text("".join(json.dumps(row)+"\n" for row in receipt["accepted"] if row["split"] == split))
    (args.out / "ingestion.json").write_text(json.dumps(receipt,indent=2)+"\n")
    validate_composition(args.out,receipt)
    print(json.dumps({"base_scenes":len(base["accepted"]),"teacher_scenes":len(teacher["accepted"]),
        "validation_test_unchanged":True}),flush=True)


if __name__ == "__main__":
    main()
