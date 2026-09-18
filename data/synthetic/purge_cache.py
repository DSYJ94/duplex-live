"""Purge bad clips from the render clip-cache (closes the cache loop).

The cache memoizes <=6-word renders per (engine, voice, text) at render
time — before the filter sees them. This script reads filter reports,
finds gate-failed short utterances (tail_babble etc.), derives the same
cache key from the report's (engine, voice, ref text), and removes those
entries so the next render re-rolls them. Good clips freeze; bad keys
keep rolling until one passes.

Run after each batch's filter (the render slurm self-chains the filter,
so this is the only manual/agent step in the loop):

  python -m data.synthetic.purge_cache --cache $DL/runs/clip-cache \
      --reports $DL/runs/synth-big3/audio/filter_report.jsonl [...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.synthetic.render import SHORT_CACHE_WORDS, cache_key  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--reports", nargs="+", required=True)
    args = ap.parse_args()

    cache = Path(args.cache)
    removed = missing = 0
    for rp in args.reports:
        for line in Path(rp).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for u in (r.get("asr_detail") or {}).get("per_utterance", []):
                if "gate_fail" not in u:
                    continue
                ref = u.get("ref") or ""
                if len(ref.split()) > SHORT_CACHE_WORDS:
                    continue  # long refs are never cached
                f = cache / cache_key(u.get("voice") or "", ref,
                                      r.get("engine") or "")
                if f.exists():
                    f.unlink()
                    removed += 1
                    print(f"  purged {f.name} ({u['gate_fail'][:60]})")
                else:
                    missing += 1
    print(f"purged {removed} cache entries "
          f"({missing} failed keys not in cache — they re-roll anyway)")


if __name__ == "__main__":
    main()
