"""OpenAI-backed content-bank expansion (2026-10-06, ADR-0003 Amended A:
closed-API content permitted under the owner's enterprise agreement;
provenance recorded in the banks sidecar). Same SPEC, validation, dedup
and output format as banks_llm — only the generator is the API, so the
output is a drop-in for scenarios.py --banks. Frontier-model banks buy
content DIVERSITY per TTS dollar (richer phrasing than local re-rolls).

Run anywhere with the key (cluster login node has outbound HTTPS):

  OPENAI_API_KEY=... python -m data.synthetic.banks_openai \
      --out banks-openai.json --model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.synthetic.banks_llm import (  # noqa: E402
    SPEC, parse_items, seeds_for, validate)


def ask(session, model: str, prompt: str, key: str) -> str:
    for backoff in (0, 5, 20):
        if backoff:
            time.sleep(backoff)
        r = session.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.9},
            timeout=120)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code not in (429, 500, 502, 503):
            r.raise_for_status()
    r.raise_for_status()  # type: ignore[misc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    import requests
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set (source the key file)")
    session = requests.Session()
    rng = random.Random(args.seed)
    banks, provenance = {}, {}
    for bank, spec in SPEC.items():
        seeds = seeds_for(bank)
        example = json.dumps(seeds[:3], ensure_ascii=False, indent=1)
        prompt = (
            f"You are writing content banks for a spoken-dialogue "
            f"dataset. Write {spec['per_call']} NEW items.\n"
            f"Item requirements: {spec['rules']}.\n"
            f"Style example (tone and length only — do not copy):\n"
            f"{example}\n"
            f"Respond with ONLY a JSON array of {spec['per_call']} "
            f"items, no numbering, no commentary.")
        keep: list = []
        seen = set(map(str.lower, map(json.dumps, seeds)))
        rounds = 0
        while len(keep) < spec["target"] and rounds < 8:
            rounds += 1
            for item in parse_items(ask(session, args.model, prompt, key)):
                k = json.dumps(item, ensure_ascii=False).lower()
                if k in seen or not validate(bank, item):
                    continue
                seen.add(k)
                keep.append(item)
            print(f"  [{bank}] {len(keep)}/{spec['target']} after "
                  f"{rounds} rounds", flush=True)
        banks[bank] = keep
        provenance[bank] = {"generator": f"openai:{args.model}",
                            "n": len(keep)}
        print(f"[banks-openai] {bank}: {len(keep)} items", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"provenance": provenance, "banks": banks}, ensure_ascii=False,
        indent=1))
    print(f"[banks-openai] wrote {out}")


if __name__ == "__main__":
    main()
