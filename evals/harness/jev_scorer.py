"""Jev content-scorer for eval sessions (use case 1, ADR-0003 provenance).

Plain words: our energy metrics measure WHEN the model spoke; this adds
WHAT it meant. One batched TypeSafe request per test — parallel Nouls and
Scores over the session transcript judged against the protocol's intent.
Outputs jev_scores.json next to harness_report.json (verdicts are
advisory until thresholds are tuned on our data).

Run on the cluster login node (key at $DL/.typesafe_key):
  python -m evals.harness.jev_scorer --session $DL/runs/harness-live-0919-1243
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.harness.protocol import T_PROTOCOL  # noqa: E402

# the questions the T1 protocol asks (from the bake-off script)
T1_QUESTIONS = [
    "What is the capital of France?",
    "How many planets are in the solar system?",
    "What is seventeen times six?",
    "Who wrote Romeo and Juliet?",
    "What colour do you get mixing blue and yellow?",
    "How do you say thank you in Japanese?",
    "What is the boiling point of water in Fahrenheit?",
    "Name three primary colours.",
    "What year did the Apollo eleven mission land on the Moon?",
    "What is the square root of one hundred and forty four?",
]


def transcript(session: Path, test: str) -> str:
    """Model-side text from a session dir (log.jsonl 'text' fields)."""
    logp = session / test / "log.jsonl"
    if not logp.exists():
        return ""
    return " ".join(json.loads(l).get("text", "").strip()
                    for l in logp.read_text().splitlines() if l.strip())


def questions_for(test: str, text: str) -> dict:
    """Parallel typed questions over shared state, per protocol intent."""
    if test == "t1":
        return {f"answers_q{i}": {
            "type": "noul",
            "instructions": f"Does `output` contain a direct answer to the "
                            f"question `questions[{i}]`?"}
                for i in range(len(T1_QUESTIONS))}
    if test == "t2":
        return {
            "answers_interrupt": {
                "type": "noul",
                "instructions": "Does `output` contain an answer to the "
                                "interruption question `interrupt`, asked "
                                "after the model was stopped mid-story?"},
            "revised_quality": {
                "type": "score",
                "instructions": "How good is the model's revised answer to "
                                "the interruption, as an answer to "
                                "`interrupt`?",
                "criteria": [
                    "Off-topic or ignores what was asked",
                    "Partially relevant but incomplete or rambling",
                    "A correct, concise, direct answer",
                ]},
        }
    if test == "t3":
        return {"continued_story": {
            "type": "noul",
            "instructions": "Does `output` show the assistant continuing "
                            "its storytelling despite the listener's short "
                            "backchannel remarks?"}}
    if test == "t4":
        return {"waited_for_full_request": {
            "type": "noul",
            "instructions": "Does `output` give the TCP-versus-UDP "
                            "explanation only in response to the complete "
                            "two-part request, rather than answering the "
                            "trailing unfinished first part immediately?"}}
    if test == "t5":
        return {"engaged_background": {
            "type": "noul",
            "instructions": "Does `output` respond to, comment on, or "
                            "otherwise engage with the background "
                            "conversation about a sports match (the model "
                            "was never addressed and should stay silent)?"}}
    return {}


def state_for(test: str, text: str) -> dict:
    s = {"output": text[:4000]}
    if test == "t1":
        s["questions"] = T1_QUESTIONS
    elif test == "t2":
        s["interrupt"] = ("Stop — actually, just tell me who built the "
                          "first locomotive, one sentence.")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--tests", default="t1,t2,t3,t4,t5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from typesafe_sdk import TypeSafeClient

    session = Path(args.session)
    client = TypeSafeClient(model="jev-latest")
    rows = {}
    for t in [x for x in args.tests.split(",") if x]:
        text = transcript(session, t)
        qs = questions_for(t, text)
        if not qs or not text.strip():
            rows[t] = {"skipped": "no transcript or no questions"}
            continue
        res = client.system_one(state_for(t, text), qs)
        row = {}
        for qid, q in qs.items():
            if q["type"] == "noul":
                row[qid] = round(res.nouls[qid].noul, 3)
            else:
                sc = res.scores[qid]
                row[qid] = {"score": round(sc.score, 2),
                            "confidence": round(sc.confidence, 3)}
        row["intent"] = T_PROTOCOL[t]["intent"]
        rows[t] = row
        print(f"[{t}] {row}", flush=True)

    out = Path(args.out or (session / "jev_scores.json"))
    out.write_text(json.dumps({
        "provenance": "closed-labeled(jev) per ADR-0003 — advisory verdicts",
        "rows": rows}, indent=1))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
