"""Native Qwen text routing diagnostic; no voice or frontend qualification."""
import argparse
import json
import subprocess
import time
from pathlib import Path


CASES = [
    ("greeting", "Hello, how are you?", "answer"),
    ("thanks", "Thanks, that helped.", "answer"),
    ("triangle", "How many sides does a triangle have?", "answer"),
    ("arithmetic", "What is six plus seven?", "answer"),
    ("definition", "What does evaporation mean?", "answer"),
    ("private-code", "Check the current entry code in the project records.", "delegate"),
    ("private-invoice", "Which of my invoices are overdue? Check the accounting records.", "delegate"),
    ("live-flight", "Check whether flight BA281 is delayed today.", "delegate"),
    ("live-weather", "Look up today's weather in Amsterdam.", "delegate"),
    ("document", "Read the project budget spreadsheet and calculate the remaining balance.", "delegate"),
    ("missing-it", "Book it for tomorrow.", "clarify"),
    ("missing-recipient", "Send them that thing.", "clarify"),
    ("missing-file", "Compare the two documents.", "clarify"),
    ("missing-project", "What is our project code? I have several projects.", "clarify"),
    ("missing-destination", "Find flights for next Tuesday.", "clarify"),
    ("not-request", "Don't check my balance. Just explain what a bank balance is.", "answer"),
    ("cancel", "Cancel the outstanding invoice lookup. Do not send its result to me.", "cancel"),
    ("corrected-lookup", "Check Thursday's room availability, not Friday's. The building is Cedar House.", "delegate"),
    ("context-clarify", "Earlier I said I have projects Birch and Cedar. Which one did you mean by our project?", "clarify"),
    ("claim-after-check", "Check the tool's booking status before you say the booking succeeded.", "delegate"),
]


def main():
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from evals.moshi_model_download import sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(CASES) != 20 or len({c[0] for c in CASES}) != 20:
        raise RuntimeError("Routing diagnostic identities changed")
    args.out.mkdir(parents=True, exist_ok=False)
    model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_id,
        dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa",
        experts_implementation="batched_mm", local_files_only=True).eval()
    model.disable_talker()
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_id, local_files_only=True)
    instructions = (
        "You are a request router for a voice assistant. Return only JSON with route and reason. "
        "The allowed routes are answer, clarify, delegate, cancel. Answer simple conversation and general knowledge. "
        "Delegate requests requiring private records, external actions, live information or document computation. "
        "Clarify when required identifying details are missing; do not invent them. Cancel when the user cancels ongoing work. "
        "Honor negation and corrections. No work has run yet, so never claim completion. Do not include hidden reasoning. /no_think")
    results = []
    with (args.out / "results.jsonl").open("w") as stream:
        for identity, request, expected in CASES:
            messages = [{"role": "system", "content": [{"type": "text", "text": instructions}]},
                        {"role": "user", "content": [{"type": "text", "text": request}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)
            inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(model.device)
            started = time.monotonic()
            with torch.inference_mode():
                tokens = model.generate(**inputs, thinker_max_new_tokens=256,
                                        thinker_do_sample=False, return_audio=False)
            torch.cuda.synchronize()
            text = processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
            try:
                response = json.loads(text.strip())
                selected = response.get("route")
                valid = selected in {"answer", "clarify", "delegate", "cancel"} and isinstance(response.get("reason"), str)
            except (ValueError, AttributeError):
                selected, valid = None, False
            row = {"id": identity, "request": request, "expected_policy_route": expected,
                "selected_route": selected, "valid_json": valid, "policy_match": valid and selected == expected,
                "generated_text": text, "seconds": time.monotonic() - started}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            results.append(row)
    report = {"model": model_id, "model_commit": getattr(model.config, "_commit_hash", None),
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "probe_source_sha256": sha256(__file__), "cases": len(results),
        "generation_path": "native model.generate, return_audio=False, default thinker EOS 151645",
        "policy_matches": sum(r["policy_match"] for r in results),
        "invalid_outputs": sum(not r["valid_json"] for r in results),
        "missed_required_delegation": sum(r["expected_policy_route"] == "delegate" and r["selected_route"] != "delegate" for r in results),
        "unnecessary_delegation": sum(r["expected_policy_route"] == "answer" and r["selected_route"] == "delegate" for r in results),
        "mode": "prompted text router, manually specified development policy",
        "fresh_voice_holdout": False, "speech_model_delegation_tested": False,
        "complete_task_success_tested": False, "training_performed": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
