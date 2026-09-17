"""Open-LLM content-bank expansion for the synthetic factory (Issue #3).

ADR-0002: open models only — Qwen3-Omni-30B-A3B thinker (Apache-2.0, staged
on Isambard) generates template-constrained variants of every hand-written
bank in scenarios.py. Seeds stay canonical in-repo; expansions land as a
banks.json sidecar consumed via `scenarios.py --banks` (never overwritten).

Audit loop before hours-scale: render a smoke batch with the expanded banks
(synth-render.slurm OUT_BASE=runs/banks-smoke), filter it, rsync, audition
at http://localhost:8765 (viz.listen).

Run on Isambard under the pylibs overlay (see banks-expand.slurm):

  python -m data.synthetic.banks_llm --out runs/banks/banks.json --per-bank 40
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# bank -> (generator prompt rules, few-shot seed lines, item shape check).
# Char bounds mirror the hand-written seeds; openers must be >=4 words
# (1-2-word prompts make the TTS babble — measured, see SILENCE_SETS note).
SPEC: dict[str, dict] = {
    "EXPLAIN": {
        "target": 24, "per_call": 6,
        "rules": "a plain-English explanation passage of ONE everyday or "
                 "technical concept, 300-500 characters, 3-5 sentences, no "
                 "headings, no lists, no quotes, no emoji, ends with a period",
        "bounds": (260, 560), "shape": "str",
    },
    "STORIES": {
        "target": 24, "per_call": 6,
        "rules": "a warm micro-story with a named person and a small "
                 "surprise, 200-360 characters, 3-4 sentences, present or "
                 "past tense, ends with a period",
        "bounds": (160, 420), "shape": "str",
    },
    "QUESTIONS_HARD": {
        "target": 18, "per_call": 6,
        "rules": "a single hard technical question demanding a trade-off "
                 "analysis, 120-280 characters, ends with a question mark",
        "bounds": (90, 320), "shape": "str",
    },
    "BACKCHANNELS": {
        "target": 12, "per_call": 10,
        "rules": "a single spoken backchannel interjection, 2-10 characters, "
                 "letters and hyphens only (like mm-hm, right, got it)",
        "bounds": (2, 12), "shape": "str",
    },
    "BARGE_INS": {
        "target": 18, "per_call": 5,
        "rules": "a JSON array of PAIRS [barge_in, yield]: barge_in is what "
                 "an interrupting user says mid-story (25-90 chars); yield "
                 "is the assistant's clean, story-agnostic answer to THAT "
                 "interruption (15-130 chars) — the yield must respond to "
                 "the barge_in, never reference a specific story",
        "bounds": None, "shape": "pair",
    },
    "TIMED_TOPICS": {
        "target": 24, "per_call": 4,
        "rules": "a JSON array of 4-item arrays [name, request, "
                 "interval_ms, callouts]: name is 1-2 words; request asks "
                 "the assistant to keep time and MUST name the same cadence "
                 "and count as the callouts (50-170 chars); interval_ms is "
                 "one of 5000, 8000, 10000, 15000, 20000, 30000, 60000; "
                 "callouts is 3-5 spoken callout strings, each under 40 "
                 "characters, matching the named cadence/count",
        "bounds": None, "shape": "timed",
    },
    "ADDRESSEE_QA": {
        "target": 20, "per_call": 8,
        "rules": "a JSON array of PAIRS [question, answer]: a short factual "
                 "question a user asks while background conversation is "
                 "happening (20-80 chars, ends with ?); a correct concise "
                 "answer (15-120 chars)",
        "bounds": None, "shape": "pair",
    },
    "SILENCE_SETS": {
        "target": 16, "per_call": 4,
        "rules": "a JSON array of 4-item arrays [answer, followup, opener, "
                 "completion]: a short assistant answer (25-60 chars); a "
                 "user follow-up question (15-50 chars, ends with ?); a "
                 "thinking opener of AT LEAST 4 words like 'Well, let me "
                 "see,' (15-40 chars, no trailing period); the completed "
                 "thought (20-80 chars)",
        "bounds": None, "shape": "quad",
    },
    "SIMULTANEOUS_SETS": {
        "target": 16, "per_call": 4,
        "rules": "a JSON array of 5-item arrays [open, main, overlap, "
                 "again, close] for a speech-practice duet: open sets up "
                 "the exercise (30-70 chars); main is the user practising "
                 "(25-70); overlap is the assistant coaching WHILE the user "
                 "is still speaking (25-90); again is the user's retry "
                 "(15-50); close is brief praise (15-60). English words "
                 "only — no spelled-out syllable drills (they do not "
                 "survive ASR filtering)",
        "bounds": None, "shape": "quint",
    },
    "DELEGATION_STALLS": {
        "target": 16, "per_call": 8,
        "rules": "one sentence an assistant says to buy time on a hard "
                 "question, polite and specific, 40-120 characters, ends "
                 "with a period",
        "bounds": (35, 140), "shape": "str",
    },
    "DELEGATION_WEAVES": {
        "target": 16, "per_call": 8,
        "rules": "one sentence an assistant uses to re-integrate a slow "
                 "background answer into the conversation, headline-first, "
                 "60-160 characters, ends with a period",
        "bounds": (55, 180), "shape": "str",
    },
}

ALLOWED_INTERVALS = {5000, 8000, 10000, 15000, 20000, 30000, 60000}


def seeds_for(bank: str) -> list:
    import data.synthetic.scenarios as sc
    return {
        "EXPLAIN": sc.EXPLAIN, "STORIES": sc.STORIES,
        "QUESTIONS_HARD": sc.QUESTIONS_HARD, "BACKCHANNELS": sc.BACKCHANNELS,
        "BARGE_INS": sc.BARGE_INS, "TIMED_TOPICS":
            [[n, r, i, c] for n, r, i, c in sc.TIMED_TOPICS],
        "ADDRESSEE_QA": [list(x) for x in sc.ADDRESSEE_QA],
        "SILENCE_SETS": [list(x) for x in sc.SILENCE_SETS],
        "SIMULTANEOUS_SETS": [list(x) for x in sc.SIMULTANEOUS_SETS],
        "DELEGATION_STALLS": sc.DELEGATION_STALLS,
        "DELEGATION_WEAVES": sc.DELEGATION_WEAVES,
    }[bank]


# ---------------------------------------------------------------- validation

_OK_CHARS = re.compile(r"^[A-Za-z0-9 ,.'%$\-:;?!]+$")


def _text_ok(s, lo, hi) -> bool:
    return (isinstance(s, str) and lo <= len(s.strip()) <= hi
            and "\n" not in s and _OK_CHARS.match(s.strip()) is not None)


def validate(bank: str, item) -> bool:
    spec = SPEC[bank]
    sh, b = spec["shape"], spec["bounds"]
    if sh == "str":
        return _text_ok(item, *b)
    if not isinstance(item, list):
        return False
    if sh == "pair":  # BARGE_INS, ADDRESSEE_QA
        return (len(item) == 2 and _text_ok(item[0], 20, 100)
                and _text_ok(item[1], 12, 140))
    if sh == "timed":  # TIMED_TOPICS
        if len(item) != 4 or not isinstance(item[3], list):
            return False
        name, req, interval, callouts = item
        return (_text_ok(name, 2, 22) and _text_ok(req, 45, 180)
                and interval in ALLOWED_INTERVALS and 3 <= len(callouts) <= 5
                and all(_text_ok(c, 2, 44) for c in callouts))
    if sh == "quad":  # SILENCE_SETS — opener >=4 words (TTS babble guard)
        return (len(item) == 4 and _text_ok(item[0], 20, 70)
                and _text_ok(item[1], 12, 60) and item[1].strip().endswith("?")
                and _text_ok(item[2], 14, 50) and len(item[2].split()) >= 4
                and _text_ok(item[3], 16, 90))
    if sh == "quint":  # SIMULTANEOUS_SETS
        return (len(item) == 5 and _text_ok(item[0], 24, 80)
                and _text_ok(item[1], 20, 80) and _text_ok(item[2], 20, 100)
                and _text_ok(item[3], 12, 60) and _text_ok(item[4], 12, 70))
    return False


# ---------------------------------------------------------------- generation

def load_model():
    import torch
    from transformers import AutoProcessor, Qwen3OmniMoeThinkerForConditionalGeneration

    model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype="bfloat16", device_map="auto",
        attn_implementation="sdpa", experts_implementation="batched_mm",
    )
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    return model, proc


def chat_batch(model, proc, prompt: str, batch: int = 4,
               max_new_tokens: int = 1600) -> list[str]:
    """Same prompt, batched rows (do_sample makes each row different).
    Single-stream 30B-A3B runs ~8 tok/s on GH200 — batching is the only
    way expansion fits a wall-clock budget (first run: 2.5 banks in 2 h
    unbatched)."""
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    try:
        text = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False, enable_thinking=False)
    except TypeError:  # template without the kwarg
        text = proc.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=False)
    tok = proc.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"
    try:
        inputs = proc(text=[text] * batch, return_tensors="pt",
                      padding=True).to(model.device)
        out = model.generate(**inputs, do_sample=True, temperature=0.9,
                             top_p=0.95, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.pad_token_id)
    finally:
        tok.padding_side = old_side
    gens = out[:, inputs["input_ids"].shape[1]:]
    strips = []
    for g in gens:
        s = proc.decode(g, skip_special_tokens=True)
        strips.append(re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip())
    return strips


def parse_items(raw: str) -> list:
    """Pull the JSON array out of a completion (models add prose)."""
    m = re.search(r"\[.*\]", raw, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def expand_bank(model, proc, bank: str, rng: random.Random,
                initial: list | None = None) -> list:
    spec = SPEC[bank]
    seeds = seeds_for(bank)
    example = json.dumps(seeds[:3] if len(seeds) >= 3 else seeds,
                         ensure_ascii=False, indent=1)
    prompt = (
        f"You are writing content banks for a spoken-dialogue dataset. "
        f"Write {spec['per_call']} NEW items.\n"
        f"Item requirements: {spec['rules']}.\n"
        f"Style example (for tone and length only — do not copy them):\n"
        f"{example}\n"
        f"Respond with ONLY a JSON array of {spec['per_call']} items, "
        f"no numbering, no commentary.")
    keep = list(initial or [])
    seen = set(map(str.lower, map(json.dumps, seeds)))
    for it in keep:
        seen.add(json.dumps(it, ensure_ascii=False).lower())
    rounds = 0
    max_rounds = 8
    while len(keep) < spec["target"] and rounds < max_rounds:
        rounds += 1
        for raw in chat_batch(model, proc, prompt):
            for item in parse_items(raw):
                key = json.dumps(item, ensure_ascii=False).lower()
                if key in seen or not validate(bank, item):
                    continue
                seen.add(key)
                keep.append(item)
        print(f"  [{bank}] {len(keep)}/{spec['target']} after {rounds} "
              f"batched rounds", flush=True)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--banks", default=None, help="comma list (default all)")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    rng = random.Random(args.seed)
    model, proc = load_model()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    provenance = {
        "model": MODEL_ID, "license": "Apache-2.0 (ADR-0002: open only)",
        "seed": args.seed, "generated": time.strftime("%Y-%m-%d"),
    }

    # resume support: existing banks under the same seed are kept, so a
    # SLURM timeout mid-run costs only the bank in flight
    banks: dict = {}
    if out.exists():
        prev = json.loads(out.read_text())
        if prev.get("provenance", {}).get("seed") == args.seed:
            banks = prev.get("banks", {})
            print(f"[resume] keeping {len(banks)} banks from {out}", flush=True)

    todo = [b for b in (args.banks.split(",") if args.banks else list(SPEC))
            if len(banks.get(b, [])) < SPEC[b]["target"]]
    t0 = time.time()
    for bank in todo:
        print(f"[expand] {bank}", flush=True)
        banks[bank] = expand_bank(model, proc, bank, rng,
                                  initial=banks.get(bank))
        out.write_text(json.dumps({"provenance": provenance, "banks": banks},
                                  ensure_ascii=False, indent=1))  # checkpoint
    print(f"done in {time.time() - t0:.0f}s -> {out}")
    for k, v in banks.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
