"""Read-only CPU audit of stored native token frames; no model inference."""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_data import validate_cache
from evals.recovery_encode import native_word_timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    import torch
    import sentencepiece
    receipt = validate_cache(args.cache)
    vocab = Path(receipt["model"]["path"]) / "tokenizer_spm_32k_3.model"
    if sha256(vocab) != receipt["model"]["files"][vocab.name]["sha256"]:
        raise RuntimeError("Native timing vocabulary changed")
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(vocab))
    results = []
    for row in receipt["examples"]:
        sidecar = json.loads((Path(receipt["conversion"]) / row["scenario_id"] / "conversation.json").read_text())
        codes = torch.load(args.cache / row["file"],map_location="cpu",weights_only=True)["codes"]
        try:
            timing = native_word_timing(codes[0,0].tolist(),sidecar["alignments"],tokenizer.encode,receipt["frame_rate"])
            result = {"scenario_id":row["scenario_id"],"passed":True,**timing}
        except RuntimeError as error:
            result = {"scenario_id":row["scenario_id"],"passed":False,"error":str(error)}
        results.append(result)
    args.out.mkdir(parents=True,exist_ok=False)
    report = {"schema":"dl-native-code-word-clock-v1","cache":str(args.cache),
        "cache_sha256":sha256(args.cache / "cache.json"),"results":results,
        "passed":all(r["passed"] for r in results)}
    (args.out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":report["passed"],"scenes":len(results),"failures":[r for r in results if not r["passed"]],
        "max_first_word_offset_ms":max((r.get("max_first_word_offset_ms",0) for r in results),default=None)}),flush=True)
    if not report["passed"]: raise RuntimeError("Cached native words failed their actual frame-clock audit")


if __name__ == "__main__": main()
