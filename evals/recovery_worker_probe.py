"""Bounded API routing/tool diagnostic; no native speech delegation claim."""
import argparse
import json
import secrets
from pathlib import Path
from evals.recovery_worker import RecordWorld,run_worker,WorkerContractError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    cases = [
        ("simple","What is six plus seven?","answer",None),
        ("private","Check the badge phrase for project Cedar.","delegate","Cedar"),
        ("clarify","What is my badge phrase? I have several projects.","clarify",None),
        ("negation","Do not look up any records. Explain what a badge is.","answer",None),
        ("corrected","Earlier I asked for Birch. I meant Cedar instead. Check its badge phrase.","delegate","Cedar"),
        ("other-project","Check the badge phrase for project Birch.","delegate","Birch"),
        ("cancel","Cancel pending-check. Do not give me its result.","cancel",None),
        ("pending-general","While pending-check is running, how many sides does a triangle have?","answer",None),
    ]
    results = []
    for identity,request,expected,project in cases:
        if args.case and identity != args.case:
            continue
        # New nonce values per episode ensure the tool, rather than parameters,
        # is the only possible source. Project names may repeat across episodes.
        records = {p:{"badge phrase":"test "+secrets.token_hex(6)} for p in ("Cedar","Birch")}
        world = RecordWorld(records)
        world.reserve("pending-check")
        try:
            result = run_worker([{"role":"user","text":request}],world,"probe-"+identity)
        except WorkerContractError as error:
            (args.out / "failure.json").write_text(json.dumps({"error":str(error),"result":error.result},indent=2)+"\n")
            raise
        tool_ok = (len(result["receipts"]) == 1 and result["receipts"][0].get("project") == project
            and records[project]["badge phrase"] in result["content"]) if project else not any(
                r["tool"] == "lookup_record" for r in result["receipts"])
        row = {"id":identity,"request":request,"expected_route":expected,"result":result,
            "policy_match":result["route"] == expected,"tool_contract_match":tool_ok}
        results.append(row)
        (args.out / "results.jsonl").write_text("".join(json.dumps(r)+"\n" for r in results))
        print(json.dumps({"id":identity,"route":result["route"],"policy_match":row["policy_match"],"tool_contract":tool_ok}),flush=True)
    report = {"cases":len(results),"policy_matches":sum(r["policy_match"] for r in results),
        "tool_contract_matches":sum(r["tool_contract_match"] for r in results),
        "actual_record_reads":sum(len([x for x in r["result"]["receipts"] if x["tool"] == "lookup_record"]) for r in results),
        "mode":"actual API semantic worker plus episode-local read-only tools", "native_voice_routing_tested":False,
        "complete_duplex_task_success_tested":False}
    (args.out / "report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
