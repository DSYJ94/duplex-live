"""T-protocol definitions for the eval harness (Issue #6).

The T1–T6 specs mirror evals/bakeoff/minicpm_bakeoff.py exactly (stimulus
layout + intent), so historical bake-off recordings replay cleanly. Factory
scenarios (data/synthetic) provide richer stimulus sets via the same
adapter: the scenario class maps to a T-test (see FACTORY_TESTS).

Verdict thresholds encode the bake-off findings as the bar: barge yield
must beat 800 ms (PersonaPlex scripted replay measured ≤500 ms; humans
~250 ms), backchannels must NOT yield, background must not engage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# seconds; generous defaults — replay metrics refine from the wavs
T_PROTOCOL: dict[str, dict] = {
    "t1": {"intent": "baseline turn-taking: 10 questions, 15 s gaps",
           "marks": {"q": [2 + i * 15 for i in range(10)]}, "total_s": 158},
    "t2": {"intent": "barge-in: long-answer bait, interrupt at 16 s",
           "marks": {"bait": [2], "barge": [16]}, "total_s": 40},
    "t3": {"intent": "backchannels mid-answer (model must NOT yield)",
           "marks": {"bait": [2], "backchannel": [10, 16, 22]}, "total_s": 45},
    "t4": {"intent": "thinking pause: user stalls 3 s mid-request",
           "marks": {"part": [2, 9]}, "total_s": 28},
    "t5": {"intent": "background-only speech, user never addresses model",
           "marks": {"background": [2, 8]}, "total_s": 22},
    "t6": {"intent": "long session (~15 min, 30 s cadence)",
           "marks": {"turn": [2 + i * 30 for i in range(17)]}, "total_s": 527},
}

# factory scenario class -> T-test semantics (stimulus = user+bg stems)
FACTORY_TESTS = {
    "barge_in": "t2", "backchannel": "t3", "hold_floor": "t4",
    "silence_semantics": "t4", "addressee": "t5", "timed_task": "t7",
    "delegation": "t8",
}

# verdict thresholds (ms / bool)
THRESH = {
    "reply_after_barge_max_ms": 800,     # bar: PersonaPlex ≤500 scripted
    "barge_to_silence_max_ms": 800,
    "response_latency_p50_max_ms": 1500,
    "backchannel_yield_max_count": 0,
    "false_engagement_max_s": 2.0,       # T5 total output speech
    "pause_resume": True,                # T4: model resumes after own pause
}

# live-run stimulus layouts (prompt stems from the bake-off Phase A synth;
# recordings exist at runs/bakeoff/prompts on the cluster)
LIVE_TESTS = {
    "t1": ([(2 + i * 15, f"t1_q{i}") for i in range(1, 11)], 2 + 9 * 15 + 15),
    "t2": ([(2, "t2_bait"), (16, "t2_interrupt")], 40),
    "t3": ([(2, "t2_bait"), (10, "t3_bc1"), (16, "t3_bc2"), (22, "t3_bc3")], 45),
    "t4": ([(2, "t4_part1"), (9, "t4_part2")], 28),
    "t5": ([(2, "t5_bg1"), (8, "t5_bg2")], 22),
}
