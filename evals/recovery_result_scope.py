"""Select actual native target frames for result supervision, without new labels."""
import math

from evals.recovery_prepare_data import spoken_words
from evals.recovery_encode import native_word_timing


def last_utterance_frames(sidecar, text_codes, frame_rate=12.5):
    last = [e for e in sidecar["timeline"]["events"]
            if e["ch"] == "assistant" and e["kind"] == "utterance"][-1]
    return [i for i, token in enumerate(text_codes) if token >= 5
            and last["actual_start_ms"]/1000 <= i/frame_rate < last["actual_end_ms"]/1000]


def private_result_frames(sidecar, text_codes, encode_word, frame_rate=12.5):
    target = spoken_words(sidecar["timeline"]["meta"]["phrase"] or "")
    arrivals = sidecar["reference_arrivals"]
    if not target or len(arrivals) != 1 or not math.isfinite(arrivals[0]["ready_at_s"]):
        raise RuntimeError("Checked phrase requires exactly one measured result arrival")
    native_word_timing(text_codes, sidecar["alignments"], encode_word, frame_rate)
    positions = [i for i, token in enumerate(text_codes) if token >= 5]
    words, ranges, flat, owners = [], [], [], []
    offset = 0
    for word, interval, speaker in sidecar["alignments"]:
        if speaker != "SPEAKER_MAIN":
            continue
        tokens = list(encode_word(word.strip()))
        index = len(words)
        words.append((word, interval))
        ranges.append(positions[offset:offset+len(tokens)])
        offset += len(tokens)
        normalized = spoken_words(word)
        flat.extend(normalized)
        owners.extend([index]*len(normalized))
    matches = [i for i in range(len(flat)-len(target)+1) if flat[i:i+len(target)] == target]
    if not matches:
        raise RuntimeError("Measured assistant words omit the checked phrase")
    result = set()
    ready = arrivals[0]["ready_at_s"]
    for index in matches:
        first, last = owners[index], owners[index+len(target)-1]
        if ((index and owners[index-1] == first)
                or (index+len(target) < len(owners) and owners[index+len(target)] == last)):
            raise RuntimeError("Checked phrase boundary falls inside a measured word")
        for owner in range(first, last+1):
            if words[owner][1][0] < ready or any(frame/frame_rate < ready for frame in ranges[owner]):
                raise RuntimeError("Checked phrase is supervised before its result is ready")
            result.update(ranges[owner])
    return sorted(result)


def main():
    import argparse
    import json
    from pathlib import Path
    from evals.moshi_model_download import sha256
    from evals.recovery_data import validate_cache
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import torch
    import sentencepiece
    cache = validate_cache(args.cache)
    manifest = json.loads(args.manifest.read_text())["rag"]
    vocabulary = Path(manifest["path"]) / "tokenizer_spm_32k_3.model"
    if sha256(vocabulary) != manifest["files"][vocabulary.name]["sha256"]:
        raise RuntimeError("Native target vocabulary changed")
    tokenizer = sentencepiece.SentencePieceProcessor(str(vocabulary))
    results = []
    for row in cache["examples"]:
        if row["split"] == "test" or row["kind"] not in ("private", "clarify"):
            continue
        sidecar = json.loads((Path(cache["conversion"]) / row["scenario_id"] / "conversation.json").read_text())
        codes = torch.load(args.cache / row["file"], map_location="cpu", weights_only=True)["codes"][0, 0].tolist()
        frames = private_result_frames(sidecar, codes, tokenizer.encode)
        results.append({"id": row["scenario_id"], "split": row["split"], "kind": row["kind"],
                        "checked_phrase_tokens": len(frames),
                        "old_last_utterance_tokens": len(last_utterance_frames(sidecar, codes))})
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "dl-private-result-scope-audit-v1", "cache_sha256": sha256(args.cache / "cache.json"),
              "examples": results, "passed": True, "test_fold_inspected": False,
              "optimizer_steps": 0, "scope": "exact native subwords of measured phrase after actual result arrival"}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
