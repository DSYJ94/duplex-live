"""Compare generated histories before the actual worker-start frame."""
import argparse
import json
from pathlib import Path


def verify_prefix(prefix):
    base = Path(f"{prefix}-base")
    reference = Path(f"{prefix}-delayed-reference")
    report = json.loads((reference / "report.json").read_text())
    events = report["task_events"]
    starts = [e for e in events if e["event"] == "worker_started"]
    if not starts:
        raise RuntimeError("No worker start to define the causal boundary")
    histories = [[json.loads(line) for line in (path / "packets.jsonl").read_text().splitlines()]
                 for path in (base, reference)]
    start = starts[0]["at_s"]
    cutoff = next(r["chunk"] for r in histories[1] if r["input_consumed_s"] >= start - 1e-9)
    if cutoff <= 0 or any(len(rows) < cutoff for rows in histories):
        raise RuntimeError("Missing pre-worker history")
    before = [rows[:cutoff] for rows in histories]
    for rows in before:
        if any("generated_token_sha256" not in r for r in rows):
            raise RuntimeError("Missing generated-token evidence")
        if not any(r["generated_token_sha256"] is not None for r in rows):
            raise RuntimeError("No generated tokens before worker starts")
    differences = [i for i, (a, b) in enumerate(zip(*before))
                   if a["generated_token_sha256"] != b["generated_token_sha256"]]
    traced_inputs = all("user_code_sha256" in r for rows in before for r in rows)
    input_differences = [i for i, (a, b) in enumerate(zip(*before))
                         if a.get("user_code_sha256") != b.get("user_code_sha256")] if traced_inputs else None
    rng_differences = [i for i, (a, b) in enumerate(zip(*before))
                       if a.get("cuda_rng_before_step_sha256") is not None
                       and a["cuda_rng_before_step_sha256"] != b.get("cuda_rng_before_step_sha256")]
    verdict = {"prefix_equal": not differences, "boundary": "actual worker-start frame",
               "worker_start_wall_s": start, "frames": cutoff, "different_frames": differences,
               "input_codes_traced": traced_inputs, "different_input_code_frames": input_differences,
               "different_rng_frames": rng_differences,
               "note": "Does not certify spoken reference use, live latency or model quality."}
    Path(f"{prefix}-causal-prefix-audit.json").write_text(json.dumps(verdict, indent=2) + "\n")
    if differences or input_differences:
        raise RuntimeError(f"Pre-worker histories differ; first token mismatch: {differences[:1]}")
    return verdict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_prefix(args.prefix)))
