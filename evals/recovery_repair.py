"""Derive a separate fresh corpus version; never mutate stamped source data."""
import argparse
import json
import shutil
from pathlib import Path
from data.synthetic.recovery_curriculum import synthesize,assemble,MODEL
from evals.moshi_model_download import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    source_specs = args.source / "specifications.json"
    scenes = json.loads(source_specs.read_text())
    repairs = []
    for scene in scenes:
        if scene["split"] == "test" and scene["kind"] != "simple":
            original_project = scene["project"]
            scene["project"] = original_project.replace("Fir ","Maple ")
            for index,(role,text) in enumerate(scene["utterances"]):
                if original_project in text:
                    replacement = text.replace(original_project,scene["project"])
                    scene["utterances"][index] = [role,replacement]
                    repairs.append({"scene":scene["scenario_id"],"index":index,"role":role,"old":text,"new":replacement,
                        "reason":"Use a phonetically distinct project name in the same presealed test family"})
        if scene["scenario_id"] == "recovery-train-023":
            index = len(scene["utterances"])-1
            role,text = scene["utterances"][index]
            scene["utterances"][index] = [role,"That record check is cancelled."]
            repairs.append({"scene":scene["scenario_id"],"index":index,"role":role,"old":text,
                "new":scene["utterances"][index][1],"reason":"Replace misheard lookup/lookout with an unambiguous cancellation target"})
    if not 1 <= len(repairs) <= 8:
        raise RuntimeError("Repair exceeds its reviewed bounded scope")
    shutil.copytree(args.source,args.out,ignore=shutil.ignore_patterns("quality*.json*","filter_report.jsonl",
        "timing_report.jsonl","train_manifest*.jsonl"))
    (args.out / "specifications.json").write_text(json.dumps(scenes,indent=2)+"\n")
    for repair in repairs:
        scene = next(s for s in scenes if s["scenario_id"] == repair["scene"])
        path = args.out / repair["scene"] / "utterances" / f"{repair['index']:02d}_{repair['role']}.wav"
        synthesize(repair["new"],scene[repair["role"]+"_voice"],path)
        print(json.dumps({"repaired":repair["scene"],"clip":repair["index"]}),flush=True)
    rows = []
    for scene in scenes:
        receipts = {i:json.loads((args.out / scene["scenario_id"] / "utterances" / f"{i:02d}_{role}.receipt.json").read_text())
                    for i,(role,_) in enumerate(scene["utterances"])}
        rows.append(assemble(scene,args.out,receipts))
    (args.out / "manifest.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    (args.out / "generation.json").write_text(json.dumps({"verified":False,"source_model":MODEL,"scenes":len(rows),
        "audio_seconds":sum(r["duration_ms"] for r in rows)/1000,"repair_api_calls":len(repairs),
        "source_batch":str(args.source),"source_specifications_sha256":sha256(source_specs),
        "source_quality_sha256":sha256(args.source / "quality.jsonl"),"repairs":repairs,
        "folds_changed":False,"all_clips_require_reverification":True},indent=2)+"\n")


if __name__ == "__main__":
    main()
