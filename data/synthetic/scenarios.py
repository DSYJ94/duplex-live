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


# TTS realizes ~72–105 ms/char with a ~0.5–0.8 s lead-in (measured from the
# v0.1 render logs). Used ONLY to schedule subsequent same-channel starts —
# actual durations come from the renderer, and the filter's
# same_channel_collision check enforces that these estimates never overlap.
def _est_ms(text: str) -> int:
    return _q(800 + len(text) * 80)


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
    "When you type a website name, your computer first asks a DNS resolver to translate "
    "it into an address. The resolver may have the answer cached from someone else's "
    "recent visit; if not, it walks up a ladder of servers, from the root, to the "
    "domain's own name server, until one of them hands back the number your computer "
    "actually connects to.",
    "GPS works by trilateration, not triangulation. Your phone listens for the exact "
    "time signals from several satellites, compares each one against its own clock, and "
    "converts the delay into a distance. With four or more satellites the only place "
    "all those spheres intersect is where you are standing.",
    "Compound interest is snowfall that becomes its own hill. Each period's interest "
    "is added to the pot, so the next period earns on a bigger pot. The first years "
    "look boring and flat, and then the curve bends hard upward, which is why starting "
    "early matters more than starting big.",
    "An mRNA vaccine is a set of instructions, not the weapon. The molecule tells your "
    "cells to build the spike protein the virus wears on its surface, your immune "
    "system notices the stranger, rehearses against it, and keeps the blueprint in "
    "memory. The instructions degrade within days; the memory stays for years.",
    "A database index is the phone book trick. Instead of scanning every row, you keep "
    "a sorted structure where the search key tells you the page to jump to. The "
    "classic shape is a balanced tree, which stays shallow no matter how much you add, "
    "so a lookup costs a handful of hops instead of a full shelf-by-shelf search.",
    "Public key cryptography is a mailbox with two slots. You hand out the public key "
    "to anyone, and they can drop a letter in, but only your private key opens the "
    "box. Signatures flip it around: the private key seals a mark that anybody can "
    "verify with the public one, and nobody can forge.",
    "Tides are the moon pulling on the ocean, but not evenly. The water facing the "
    "moon is pulled hardest, the Earth itself is pulled medium, and the far ocean is "
    "left behind, which bulges the sea on both sides at once. The Earth rotates "
    "through those two bulges, so most coasts get two high tides a day.",
    "A jet engine is a chain of squeezes. Fans push air into compressors that squeeze "
    "it dozens of times over, fuel burns in the middle and expands violently, and the "
    "hot blast spins turbines on its way out. The turbines drive the compressors at "
    "the front, so the whole loop feeds itself.",
    "A refrigerator does not add cold, it moves heat. A fluid evaporates inside the "
    "cabinet, soaking up heat the way sweat cools skin, then a compressor squeezes "
    "the vapour outside where it condenses and dumps the heat into your kitchen. The "
    "cycle repeats, pumping warmth out of the box one loop at a time.",
    "A Fourier transform is asking a chord which notes it contains. Any wiggly signal "
    "can be rewritten as a sum of pure steady tones, and the transform hands you the "
    "recipe: which frequencies are present and how loud each one is. Engineers use it "
    "to pull a voice out of noise, or a vibration out of a healthy machine.",
    "Backpropagation is blame flowing backwards. When the network guesses wrong, you "
    "measure the error at the output, then walk layer by layer to the input, asking "
    "each connection how much it contributed. Every weight nudges itself a little "
    "against its own blame, and a million small nudges make the guess less wrong.",
    "The sky is blue because of Rayleigh scattering. Air molecules scatter short "
    "wavelengths far more than long ones, so sunlight bouncing around the atmosphere "
    "throws blue in every direction, and that scattered blue is what reaches your eye "
    "from everywhere except the sun itself. At sunset the light travels through more "
    "air, the blue is scattered away entirely, and the survivors are orange and red.",
]

BACKCHANNELS = ["mm-hm", "yeah", "right", "uh-huh", "okay", "I see", "sure", "mhm", "got it"]

QUESTIONS_HARD = [
    "Can you compare the consistency guarantees of Raft and Paxos, and tell me which "
    "one I'd want for a globally distributed ledger and why?",
    "Walk me through what happens to TCP congestion control when you have two hundred "
    "milliseconds of latency and one percent packet loss on the path.",
    "Explain the trade-offs between eventual consistency and strong consistency for a "
    "shopping cart service, with concrete failure examples.",
    "Why does attention scale better than recurrence for long sequences, and where "
    "does a recurrent design still win? Be specific about the costs.",
    "For a legal discovery pipeline, when would vector search beat keyword search, "
    "and what are the classic failure modes of each?",
    "A service has an occasional two hundred millisecond latency spike. Would you "
    "reach for metrics or distributed traces first, and why?",
    "Argue both sides: garbage collection versus reference counting for a latency "
    "sensitive voice service.",
    "How exactly does TLS one point three save a round trip in the handshake, and "
    "what security trade does zero round trip resumption make?",
    "Explain the duck curve problem for grids with lots of solar, and rank the "
    "realistic mitigations.",
    "In-context learning versus fine-tuning a small model: compare cost, quality, "
    "and staleness, and say when each one wins.",
    "What actually goes wrong inside a distributed message queue during a network "
    "partition, and how do the major designs cope with it?",
    "How does the choice of window function cause spectral leakage in a Fourier "
    "transform, and how should I pick one for a vibration analysis?",
]

# (name, user request, callout interval ms, callout texts) — request and callouts
# must agree on cadence and count (the v0 pomodoro promised five minutes, delivered
# three callouts; fixed)
TIMED_TOPICS = [
    ("pomodoro", "I want to do a short pomodoro. Tell me when each minute passes, "
     "three minutes, starting now.", 60000,
     ["1 minute.", "2 minutes.", "3 minutes."]),
    ("plank", "I'm holding a plank. Count out every ten seconds for me, three times.",
     10000, ["10 seconds.", "20 seconds.", "30 seconds."]),
    ("wall sit", "Wall sit time. Call out every fifteen seconds, three rounds please.",
     15000, ["15 seconds.", "30 seconds.", "45 seconds."]),
    ("box breathing", "Walk me through four rounds of box breathing, calling each "
     "round as it starts.", 8000,
     ["Round one, breathe in.", "Round two, breathe in.", "Round three, breathe in.",
      "Round four, breathe in."]),
    ("countdown", "Count down from thirty in steps of five with me, one step every "
     "five seconds.", 5000,
     ["Twenty five.", "Twenty.", "Fifteen.", "Ten.", "Five."]),
    ("intervals", "I'm doing run intervals. Tell me at every thirty seconds, "
     "four times.", 30000,
     ["Thirty seconds.", "One minute.", "Ninety seconds.", "Two minutes."]),
    ("hamstring stretch", "I'm holding a hamstring stretch. Count each twenty "
     "seconds, three times.", 20000,
     ["20 seconds.", "40 seconds.", "60 seconds."]),
    ("push ups", "I'm doing push-ups. Give me a rep target every ten seconds, "
     "three times.", 10000,
     ["Ten reps.", "Twenty reps.", "Thirty reps."]),
]

STORIES = [
    "Once upon a time there was a lighthouse keeper who collected messages in bottles. "
    "Every morning she walked the beach, and every morning the tide left her exactly one. "
    "She answered every single letter, even the ones that were only a drawing of a whale.",
    "My grandfather had a workshop full of broken radios, and he swore each one failed "
    "in its own way. Fixing them, he said, was mostly listening. You turned the dial "
    "slowly and the radio told you where it hurt.",
    "There was a mapmaker who named every unnamed street after a dog she had met. The "
    "town council objected, but by then three thousand people were giving directions "
    "by dogs, and nobody could bear to undo it.",
    "The night baker left one loaf on the sill every Friday for the inspector who "
    "never came. When she finally retired, the town confessed: there had never been "
    "an inspector. From then on they left the loaf for her.",
    "A ferry dog rode the crossing four times a day for eleven years and refused to "
    "miss a sailing. On the rare days the boat stayed in port, he sat at the gangway "
    "at departure time anyway, watching the empty water like a man keeping an "
    "appointment.",
    "The clock tower keeper noticed the great clock ran slow in hot weather, so every "
    "summer he climbed up and lifted the pendulum a hair. Fifty years of tiny "
    "corrections, logged in a notebook, and the town never once rang in the new year "
    "late.",
    "In the seed vault they say the quietest room is the one where nothing is "
    "growing. The librarian disagreed. She said every jar in the dark was a field "
    "holding its breath, waiting for its turn under a sky.",
    "The woman on the cliff counted storms the way other people count sheep. Forty "
    "one, forty two, she'd say, and mark the sky with a pencil. When the great storm "
    "came she counted that one too, and then, for the first time in her life, she "
    "slept through the night.",
    "The bridge painter started at the north end on his first day and retired at the "
    "south end forty years later. They gave him a medal, and he said the bridge had "
    "given him better, a reason to be somewhere high every single morning.",
    "The cable layers said you could hear the ocean thinking through the copper. On "
    "quiet shifts the old hands pressed their ears to the sheathing and argued about "
    "what it meant, and the newest hire, who had heard it too, never said a word.",
]

# (barge-in text, clean-yield response) — paired so the yield always answers the
# actual interruption; yields are story-agnostic so any STORIES draw fits
BARGE_INS = [
    ("Wait wait wait, stop, actually, can you make it shorter?",
     "Sure, no problem. Short version: it all works out in the end, promise."),
    ("Hold on, sorry, quick question first: what time is it?",
     "Of course. It's just past four by my clock. Shall I pick the story back up?"),
    ("Actually stop. I've changed my mind, tell me a joke instead.",
     "No problem. Why did the radio go to therapy? Too much static with its past."),
    ("Sorry to cut in, you're going too fast, can you slow down?",
     "My fault entirely. Let me back up a little and take it slower."),
    ("Stop one second. Is any of this actually important?",
     "Fair question. The middle part matters, the rest is just colour."),
    ("Wait, before you continue. Who is this about again?",
     "The keeper, the one at the start. Let me restart from her morning."),
    ("Hmm, actually skip ahead, just give me the ending.",
     "Okay, ending only: she never stops, and the town finally notices."),
    ("Quick pause, someone's at my door, one second.",
     "Take your time. I'll pick it up right here when you're back."),
    ("No no, stop, you've lost me. Can you start over?",
     "Sure thing. From the top, and simpler this time."),
]

DELEGATION_STALLS = [
    "Good question, let me think that through properly, give me a few seconds.",
    "That deserves a careful answer. Bear with me for a moment.",
    "Let me check my working on that, one moment.",
    "I want to get this right rather than fast. Two seconds.",
    "Hold on, that one's subtle. Let me reason it through.",
]

DELEGATION_WEAVES = [
    "Alright. The short answer is: it depends on your failure model, but here's the "
    "reasoning, step by step.",
    "Okay, here's the careful version. The headline first, then the reasoning behind it.",
    "Right, I've got it. Short answer first, then I'll qualify it properly.",
    "Done thinking. There is a clean answer, but it comes with one big caveat.",
]


# ---------------------------------------------------------------- generators

ADDRESSEE_QA = [
    ("Hey, what's the capital of Australia?",
     "It's Canberra. Not Sydney, which surprises most people."),
    ("Quick one, how many continents are there?",
     "Seven, by the usual count."),
    ("Hey, what's the chemical symbol for iron?",
     "Fe, from the Latin ferrum."),
]


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
    t += _est_ms(text[:cut]) + _q(pause)
    tl.add(Event(A, t, text[cut:], "utterance", note=f"resume after {pause}ms pause"))
    tl.meta["pause_ms"] = pause
    return tl


def gen_addressee(rng: random.Random, idx: int) -> Timeline:
    """T5: background speech not addressed to the assistant."""
    tl = Timeline(f"addressee-{idx:03d}", "addressee")
    q, a = rng.choice(ADDRESSEE_QA)
    tl.add(Event(U, 0, q, "utterance"))
    bg1 = rng.choice(STORIES)[:120]
    tl.add(Event(B, 800, bg1, "utterance", gain=0.5,
                  note="background speaker, not addressed to assistant"))
    tl.add(Event(A, _q(3600), a, "utterance"))
    bg2 = rng.choice(STORIES)[120:260]
    tl.add(Event(B, 800 + _est_ms(bg1) + 800, bg2, "utterance", gain=0.5))
    return tl


def gen_backchannel(rng: random.Random, idx: int) -> Timeline:
    """T3: user backchannels while assistant explains; assistant must not yield."""
    tl = Timeline(f"backchannel-{idx:03d}", "backchannel")
    req = rng.choice(["Tell me how this works.", "Walk me through it.",
                      "How does that work, then?"])
    tl.add(Event(U, 0, req, "utterance"))
    t = _est_ms(req) + 800
    text = rng.choice(EXPLAIN)
    words = text.split()
    third = len(words) // 3
    for i, chunk in enumerate([" ".join(words[:third]),
                               " ".join(words[third:2 * third]),
                               " ".join(words[2 * third:])]):
        tl.add(Event(A, t, chunk, "utterance"))
        bc_start = t + rng.randint(800, 2000)
        tl.add(Event(U, bc_start, rng.choice(BACKCHANNELS), "backchannel", gain=0.7))
        t += _est_ms(chunk) + 500
    return tl


def gen_barge_in(rng: random.Random, idx: int) -> Timeline:
    """T2 + retraction: user interrupts mid-assistant-speech; assistant must yield."""
    tl = Timeline(f"barge_in-{idx:03d}", "barge_in")
    tl.add(Event(U, 0, rng.choice([
        "Tell me a story.", "Tell me something interesting.",
        "Go on then, tell me a tale.", "Distract me with a story.",
    ]), "utterance"))
    t = 2000
    story = rng.choice(STORIES)
    tl.add(Event(A, t, story, "utterance", note="will be cut off"))
    barge_at = t + rng.randint(2500, 4500)
    barge, yield_text = rng.choice(BARGE_INS)
    tl.add(Event(U, barge_at, barge, "barge_in"))
    tl.add(Event(A, barge_at + _q(1500), yield_text, "utterance",
                 note="clean yield + revised response"))
    tl.meta["barge_at_ms"] = barge_at
    return tl


def gen_timed_task(rng: random.Random, idx: int) -> Timeline:
    """Time-aware task: pacing callouts on a schedule."""
    tl = Timeline(f"timed_task-{idx:03d}", "timed_task")
    name, req, interval, callouts = rng.choice(TIMED_TOPICS)
    tl.add(Event(U, 0, req, "utterance"))
    ack = f"Starting the {name} now. I'll keep time."
    t_ack = _est_ms(req) + 500
    tl.add(Event(A, t_ack, ack, "utterance"))
    t = t_ack + _est_ms(ack) + 800
    for c in callouts:
        tl.add(Event(A, t, c, "utterance"))
        t += interval
    tl.add(Event(U, t, rng.choice([
        "Okay, stop, thanks.", "That's enough, thank you.", "Alright, I'm done, stop.",
    ]), "utterance"))
    tl.meta["interval_ms"] = interval
    return tl


def gen_delegation(rng: random.Random, idx: int) -> Timeline:
    """Delegation: hard question -> graceful stall -> woven result."""
    tl = Timeline(f"delegation-{idx:03d}", "delegation")
    q = rng.choice(QUESTIONS_HARD)
    stall = rng.choice(DELEGATION_STALLS)
    tl.add(Event(U, 0, q, "utterance"))
    t_stall = _est_ms(q) + 500
    tl.add(Event(A, t_stall, stall, "utterance", note="stall while delegating"))
    tl.add(Event("assistant", t_stall, "", "marker", note="<delegate> boundary here"))
    tl.add(Event(A, t_stall + _est_ms(stall) + 5000, rng.choice(DELEGATION_WEAVES),
                 "utterance", note="re-integrated delegation result"))
    return tl


SILENCE_SETS = [
    # (end-of-thought answer, follow-up, thinking opener, completion) — the pause
    # after the opener means "I'm composing", the pause after the answer means
    # "done". Openers are 4+ words: 1-2-word prompts make the TTS babble
    # (measured: "Well," -> 3.1 s + drift, "Let me think." -> 4.5 s).
    ("So the answer is forty two.", "Okay. And why is that?",
     "Well, let me see,",
     "because the computation was designed that way."),
    ("The train leaves at six, not seven.", "Six? Are you sure about that?",
     "Let me think that through,",
     "yes, six. The timetable changed back in May."),
    ("It costs about three hundred dollars.", "Three hundred? For that?",
     "Hang on, let me check,",
     "I said about. The base model is two fifty."),
]


def gen_silence_semantics(rng: random.Random, idx: int) -> Timeline:
    """Silence semantics: same-length pauses with different meaning."""
    tl = Timeline(f"silence-{idx:03d}", "silence_semantics")
    ans, follow, opener, done = rng.choice(SILENCE_SETS)
    tl.add(Event(A, 0, ans, "utterance",
                 note="end-of-thought: silence after = turn complete"))
    t_follow = _q(rng.randint(4000, 6500))
    tl.add(Event(U, t_follow, follow, "utterance"))
    t_open = t_follow + _est_ms(follow) + _q(rng.randint(1500, 3000))
    tl.add(Event(A, t_open, opener, "utterance", note="inviting/thinking pause follows"))
    tl.add(Event(A, t_open + _est_ms(opener) + _q(rng.randint(600, 1500)), done,
                 "utterance"))
    return tl


SIMULTANEOUS_SETS = [
    # (open, user_main, assistant_overlap, user_again, close) — assistant speech
    # genuinely overlaps the user's, same channel-less duet
    ("Listen to my pronunciation: 'anemone'.",
     "Anemone. A-nem-o-ne.",
     "Good, almost, stress the second syllable, a-NEM-o-nee.",
     "A-NEM-o-nee.", "There you go, that's it."),
    ("I'll count to five, jump in with me when I slow down.",
     "One, two, three, four, five.",
     "Three, four, five. Good steady pace, keep it going.",
     "Six, seven, eight, nine, ten.",
     "Perfect, nice and even all the way up."),
    ("Read the next line along with me when I get going.",
     "She walked the beach every single morning.",
     "Every morning, and the tide left her exactly one bottle.",
     "Exactly one bottle, every time.",
     "Great, you're tracking the line perfectly."),
]


def gen_simultaneous(rng: random.Random, idx: int) -> Timeline:
    """Simultaneous speech: live commentary / correction / chorus overlap.

    Cross-channel overlap is the point; same-channel starts are scheduled
    with _est_ms so consecutive events never collide.
    """
    tl = Timeline(f"simultaneous-{idx:03d}", "simultaneous")
    open_, main, overlap, again, close = rng.choice(SIMULTANEOUS_SETS)
    t_main = _est_ms(open_) + _q(rng.randint(200, 800))
    t_ov = t_main + _q(rng.randint(800, 1600))  # assistant joins mid-utterance
    t_again = max(t_ov + _est_ms(overlap) - _q(1200),        # overlap the tail
                  t_main + _est_ms(main) + 400)               # but not user-on-user
    t_close = t_again + _est_ms(again) + _q(rng.randint(400, 1000))
    tl.add(Event(U, 0, open_, "utterance"))
    tl.add(Event(U, t_main, main, "utterance", note="user speaking"))
    tl.add(Event(A, t_ov, overlap, "utterance", note="overlapping commentary"))
    tl.add(Event(U, t_again, again, "utterance"))
    tl.add(Event(A, t_close, close, "utterance", gain=0.9))
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
