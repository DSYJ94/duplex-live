"""Scenario generators for the synthetic interaction-dynamics factory (Issue #3).

Each generator returns a Timeline covering one scenario class from
docs/02-data.md Layer 3. Content banks are hand-written in-repo (no closed-API
outputs anywhere in the pipeline — ADR-0002).

Usage:  python -m data.synthetic.scenarios --out timelines/ --per-class 2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from .timeline import Event, GRID_MS, Timeline

U, A, B = "user", "assistant", "background"


def _q(ms: int) -> int:
    return round(ms / GRID_MS) * GRID_MS


# ---------------------------------------------------------------- content banks

EXPLAIN = [
    "So the way a hash map works is that each key gets run through a hash function, "
    "which turns it into a number, and that number tells us which bucket to look in. "
    "When two keys land in the same bucket we call that a collision, and the usual fix "
    "is to keep a small chain of entries inside each bucket and walk it until we find "
    "the exact key we asked for.",
    "Photosynthesis is basically the plant running solar power. Light hits chlorophyll "
    "in the leaf, that energy splits water molecules, and the plant uses the freed "
    "electrons to turn carbon dioxide into sugar. Oxygen is just the exhaust gas of "
    "that whole assembly line.",
    "A transformer model reads the whole sentence at once instead of word by word. "
    "Every word looks at every other word through attention scores, figures out which "
    "ones matter for understanding it, and blends their meanings together. Do that "
    "dozens of times in layers and you get a pretty rich understanding of the text.",
]

BACKCHANNELS = ["mm-hm", "yeah", "right", "uh-huh", "okay", "I see"]

QUESTIONS_HARD = [
    "Can you compare the consistency guarantees of Raft and Paxos, and tell me which "
    "one I'd want for a globally distributed ledger and why?",
    "Walk me through what happens to TCP congestion control when you have two hundred "
    "milliseconds of latency and one percent packet loss on the path.",
    "Explain the trade-offs between eventual consistency and strong consistency for a "
    "shopping cart service, with concrete failure examples.",
]

TIMED_TOPICS = [
    ("pomodoro", "I want to do a pomodoro session. Tell me when each minute passes, "
     "for five minutes, starting now."),
    ("plank", "I'm holding a plank. Count out every ten seconds for me until I say stop."),
]

STORIES = [
    "Once upon a time there was a lighthouse keeper who collected messages in bottles. "
    "Every morning she walked the beach, and every morning the tide left her exactly one. "
    "She answered every single letter, even the ones that were only a drawing of a whale.",
    "My grandfather had a workshop full of broken radios, and he swore each one failed "
    "in its own way. Fixing them, he said, was mostly listening. You turned the dial "
    "slowly and the radio told you where it hurt.",
]


# ---------------------------------------------------------------- generators

def gen_hold_floor(rng: random.Random, idx: int) -> Timeline:
    """T4: assistant mid-thought pause (2–4 s). Correct behavior: user waits."""
    tl = Timeline(f"hold_floor-{idx:03d}", "hold_floor")
    t = 0
    tl.add(Event(U, t, "Explain this to me, take your time.", "utterance")); t += 4000
    text = rng.choice(EXPLAIN)
    half = len(text) // 2
    cut = text.rfind(" ", 0, half)
    tl.add(Event(A, t, text[:cut], "utterance"))
    pause = rng.choice([2000, 3000, 4000])
    t += _q(len(text[:cut]) * 65) + _q(pause)
    tl.add(Event(A, t, text[cut:], "utterance", note=f"resume after {pause}ms pause"))
    tl.meta["pause_ms"] = pause
    return tl


def gen_addressee(rng: random.Random, idx: int) -> Timeline:
    """T5: background speech not addressed to the assistant."""
    tl = Timeline(f"addressee-{idx:03d}", "addressee")
    tl.add(Event(U, 0, "Hey, what's the capital of Australia?", "utterance"))
    tl.add(Event(B, 800, rng.choice(STORIES)[:120], "utterance", gain=0.5,
                  note="background speaker, not addressed to assistant"))
    tl.add(Event(A, 4000, "It's Canberra. Not Sydney, which surprises most people.", "utterance"))
    tl.add(Event(B, 8000, rng.choice(STORIES)[120:260], "utterance", gain=0.5))
    return tl


def gen_backchannel(rng: random.Random, idx: int) -> Timeline:
    """T3: user backchannels while assistant explains; assistant must not yield."""
    tl = Timeline(f"backchannel-{idx:03d}", "backchannel")
    tl.add(Event(U, 0, "Tell me how this works.", "utterance"))
    t = 2500
    text = rng.choice(EXPLAIN)
    words = text.split()
    third = len(words) // 3
    for i, chunk in enumerate([" ".join(words[:third]),
                               " ".join(words[third:2 * third]),
                               " ".join(words[2 * third:])]):
        tl.add(Event(A, t, chunk, "utterance"))
        bc_start = t + rng.randint(800, 2000)
        tl.add(Event(U, bc_start, rng.choice(BACKCHANNELS), "backchannel", gain=0.7))
        t += _q(len(chunk) * 65) + 300
    return tl


def gen_barge_in(rng: random.Random, idx: int) -> Timeline:
    """T2 + retraction: user interrupts mid-assistant-speech; assistant must yield."""
    tl = Timeline(f"barge_in-{idx:03d}", "barge_in")
    tl.add(Event(U, 0, "Tell me a story.", "utterance"))
    t = 2000
    story = rng.choice(STORIES)
    tl.add(Event(A, t, story, "utterance", note="will be cut off"))
    barge_at = t + rng.randint(2500, 4500)
    tl.add(Event(U, barge_at, rng.choice([
        "Wait wait wait, stop, actually, can you make it shorter?",
        "Hold on, sorry, quick question first: what time is it?",
        "Actually stop. I've changed my mind, tell me a joke instead.",
    ]), "barge_in"))
    tl.add(Event(A, barge_at + _q(1500), "Sure, no problem. " + rng.choice([
        "Short version: she answered every letter, even the whale one.",
        "Why did the radio go to therapy? Too much static with its past.",
        "It's story time somewhere, just not here anymore.",
    ]), "utterance", note="clean yield + revised response"))
    tl.meta["barge_at_ms"] = barge_at
    return tl


def gen_timed_task(rng: random.Random, idx: int) -> Timeline:
    """Time-aware task: pacing callouts on a schedule."""
    tl = Timeline(f"timed_task-{idx:03d}", "timed_task")
    name, req = rng.choice(TIMED_TOPICS)
    tl.add(Event(U, 0, req, "utterance"))
    tl.add(Event(A, 3000, f"Starting the {name} now. I'll keep time.", "utterance"))
    t = 8000
    for i in range(1, 4):
        tl.add(Event(A, t, f"{i} minute{'s' if i > 1 else ''}." if name == "pomodoro"
                       else f"{i * 10} seconds.", "utterance"))
        t += _q(60000 if name == "pomodoro" else 10000)
    tl.add(Event(U, t, "Okay, stop, thanks.", "utterance"))
    return tl


def gen_delegation(rng: random.Random, idx: int) -> Timeline:
    """Delegation: hard question -> graceful stall -> woven result."""
    tl = Timeline(f"delegation-{idx:03d}", "delegation")
    tl.add(Event(U, 0, rng.choice(QUESTIONS_HARD), "utterance"))
    tl.add(Event(A, 4500, rng.choice([
        "Good question, let me think that through properly, give me a few seconds.",
        "That deserves a careful answer. Bear with me for a moment.",
    ]), "utterance", note="stall while delegating"))
    tl.add(Event("assistant", 4500, "", "marker", note="<delegate> boundary here"))
    tl.add(Event(A, 12000, "Alright. The short answer is: it depends on your failure "
                           "model, but here's the reasoning, step by step.", "utterance",
                 note="re-integrated delegation result"))
    return tl


def gen_silence_semantics(rng: random.Random, idx: int) -> Timeline:
    """Silence semantics: same-length pauses with different meaning."""
    tl = Timeline(f"silence-{idx:03d}", "silence_semantics")
    tl.add(Event(A, 0, "So the answer is forty two.", "utterance",
                 note="end-of-thought: silence after = turn complete"))
    tl.add(Event(U, 5000, "Okay. And why is that?", "utterance"))
    tl.add(Event(A, 9000, "Well,", "utterance", note="inviting/thinking pause follows"))
    tl.add(Event(A, 12000, "because the computation was designed that way.", "utterance"))
    return tl


def gen_simultaneous(rng: random.Random, idx: int) -> Timeline:
    """Simultaneous speech: live commentary / pronunciation correction overlap."""
    tl = Timeline(f"simultaneous-{idx:03d}", "simultaneous")
    tl.add(Event(U, 0, "Listen to my pronunciation: 'anemone'.", "utterance"))
    tl.add(Event(U, 2500, "Anemone. A-nem-o-ne.", "utterance", note="user practicing"))
    tl.add(Event(A, 3500, "Good, almost, stress the second syllable, a-NEM-o-nee.",
                  "utterance", note="overlapping correction"))
    tl.add(Event(U, 7000, "A-NEM-o-nee.", "utterance"))
    tl.add(Event(A, 8000, "There you go, that's it.", "utterance", gain=0.9))
    return tl


GENERATORS = {
    "hold_floor": gen_hold_floor,
    "addressee": gen_addressee,
    "backchannel": gen_backchannel,
    "barge_in": gen_barge_in,
    "timed_task": gen_timed_task,
    "delegation": gen_delegation,
    "silence_semantics": gen_silence_semantics,
    "simultaneous": gen_simultaneous,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="timelines")
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    n = 0
    for cls, gen in GENERATORS.items():
        for i in range(args.per_class):
            tl = gen(rng, i)
            tl.save(out / f"{tl.scenario_id}.json")
            n += 1
    print(f"wrote {n} timelines to {out}/")


if __name__ == "__main__":
    main()
