"""The 80 ms micro-turn grid (Issue #2, docs/09 §1).

Turns a factory scenario (timeline.actual.json + stem refs) into a per-tick
ledger: for every 80 ms tick, the user-in activity, the assistant emission,
speech-continuation state, and in-stream markers. This is the canonical
intermediate between the synthetic factory and the training sampler —
audio frames (AuT slots) attach to ticks by index at training time.

Emission per tick is exactly one of (docs/09 §1, as amended by
ADR-0004 / v2 label contract):
  a word        -> forwarded to the Talker (speech continues)
  <speak>       -> keep voicing: inside an utterance, no new word
  <sil>         -> no new text; Talker silent (between utterances)
  <yield>       -> retraction: flush pending codec frames (barge-in)
  <delegate>    -> async boundary fires
  <t+Ns>        -> explicit timestamp after a >= 1 s assistant gap

v0 alignment: words are distributed across an utterance's actual span
char-proportionally (forced word alignment is the documented upgrade).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tokenizer.vocab import BARGE_IN, DELEGATE, SIL, SPEAK, YIELD, timing_token  # noqa: E402

GRID_MS = 80


@dataclass
class Tick:
    idx: int
    t_ms: int
    user_in: bool = False        # user-channel audio active this tick
    emission: str = SIL          # assistant emission (word or special)
    speech_active: bool = False  # assistant speech span active (Talker voicing)
    marks: list[str] = field(default_factory=list)  # in-stream markers

    def is_default(self) -> bool:
        return (not self.user_in and self.emission == SIL
                and not self.speech_active and not self.marks)


@dataclass
class GridSequence:
    scenario_id: str
    scenario_class: str
    n_ticks: int
    ticks: list[Tick]
    audio: dict[str, str]        # channel -> stem path (AuT/Talker attach here)
    meta: dict = field(default_factory=dict)

    def stats(self) -> dict:
        n = len(self.ticks)
        return {
            "n_ticks": n,
            "duration_ms": n * GRID_MS,
            "user_ticks": sum(t.user_in for t in self.ticks),
            "speech_ticks": sum(t.speech_active for t in self.ticks),
            "overlap_ticks": sum(t.user_in and t.speech_active for t in self.ticks),
            "speak_ticks": sum(t.emission == SPEAK for t in self.ticks),
            "yield_events": sum(t.emission == YIELD for t in self.ticks),
            "delegate_events": sum(t.emission == DELEGATE for t in self.ticks),
            "words": sum(1 for t in self.ticks
                         if t.emission not in (SIL, SPEAK, YIELD, DELEGATE)
                         and not t.emission.startswith("<t+")),
        }


def _tick_range(start_ms: int, end_ms: int) -> range:
    return range(round(start_ms / GRID_MS), round(end_ms / GRID_MS))


def _place_words(words: list[str], start_ms: int, end_ms: int) -> list[tuple[int, str]]:
    """Char-proportional word placement inside a span (v0 alignment)."""
    if not words or end_ms <= start_ms:
        return []
    lens = [len(w) for w in words]
    total = sum(lens) or 1
    out, acc = [], 0
    span = round(end_ms / GRID_MS) - round(start_ms / GRID_MS)
    for w, ln in zip(words, lens):
        tick = round(start_ms / GRID_MS) + min(
            span - 1, int((acc + ln / 2) / total * span))
        out.append((tick, w))
        acc += ln
    return out


# words spoken per second estimate for truncation-aware placement (matches
# filter.MS_PER_WORD_EST = 320 ms/word)
MS_PER_WORD_EST = 320


def _word_ticks(text: str, start_ms: int, end_ms: int,
                truncated: bool) -> list[tuple[int, str]]:
    """Words placed on ticks; for truncated (cut) utterances, words beyond
    the cut were never voiced — place over the estimated FULL span and keep
    only what landed before the cut tick (retraction semantics)."""
    words = text.split()
    if not truncated:
        return _place_words(words, start_ms, end_ms)
    est_end = start_ms + max(len(words) * MS_PER_WORD_EST, end_ms - start_ms)
    cut_tick = round(end_ms / GRID_MS)
    return [(i, w) for i, w in _place_words(words, start_ms, est_end)
            if i < cut_tick]


def build_grid(tl, audio: dict[str, str] | None = None) -> GridSequence:
    """tl: data.synthetic.timeline.Timeline with actual times filled in."""
    speech = [e for e in tl.events if e.text and e.kind != "marker"]
    markers = [e for e in tl.events if e.kind == "marker"]
    end_ms = max((e.actual_end_ms or e.quantized_start() + 800) for e in tl.events)
    n_ticks = round(end_ms / GRID_MS) + 1
    ticks = [Tick(i, i * GRID_MS) for i in range(n_ticks)]

    # in-stream activity: everything the model HEARS — user utterances,
    # backchannels, barge-ins AND the background channel (addressee/T5:
    # background must be audible-not-addressed, so it rides the in-slot;
    # training mixes stem_user + stem_background as the AuT input)
    for ev in speech:
        if ev.ch == "assistant":
            continue
        for i in _tick_range(ev.actual_start_ms, ev.actual_end_ms):
            if 0 <= i < n_ticks:
                ticks[i].user_in = True

    # barge-in markers + yield emission at the interruption tick
    for ev in speech:
        if ev.kind != "barge_in":
            continue
        i = round(ev.actual_start_ms / GRID_MS)
        if 0 <= i < n_ticks:
            ticks[i].marks.append(BARGE_IN)

    # assistant emissions: explicit timestamps after long gaps
    asst = sorted((e for e in speech if e.ch == "assistant"),
                  key=lambda e: e.actual_start_ms)
    prev_end = None
    for ev in asst:
        if prev_end is not None:
            tok = timing_token(ev.actual_start_ms - prev_end)
            if tok:
                i = round(ev.actual_start_ms / GRID_MS)
                if 0 <= i < n_ticks and ticks[i].emission == SIL:
                    ticks[i].emission = tok
        prev_end = ev.actual_end_ms

    # yield at each cut: the truncated utterance's end == the barge instant
    for ev in asst:
        if "truncated at barge-in" not in (ev.note or ""):
            continue
        i = round(ev.actual_end_ms / GRID_MS)
        if 0 <= i < n_ticks:
            ticks[i].emission = YIELD

    # delegate markers
    for m in markers:
        if "delegate" in (m.note or ""):
            i = round(m.quantized_start() / GRID_MS)
            if 0 <= i < n_ticks:
                ticks[i].emission = DELEGATE

    # assistant words + speech continuation. Words outrank timing tokens
    # (auxiliary) and never collide-drop: a word whose tick is taken by
    # another word/special shifts forward to the next free tick.
    def _free(i: int) -> bool:
        return ticks[i].emission == SIL or ticks[i].emission.startswith("<t+")

    for ev in asst:
        for i in _tick_range(ev.actual_start_ms, ev.actual_end_ms):
            if 0 <= i < n_ticks:
                ticks[i].speech_active = True
        trunc = "truncated at barge-in" in (ev.note or "")
        for tick_i, w in _word_ticks(ev.text, ev.actual_start_ms,
                                     ev.actual_end_ms, trunc):
            i = tick_i
            while i < n_ticks and not _free(i):
                i += 1
            if i < n_ticks:
                ticks[i].emission = w

    # v2 label contract (ADR-0004): a tick inside an assistant utterance
    # that carries no word and no special is an explicit CONTINUE-
    # SPEAKING emission, not silence. Utterance-span ticks only — the
    # gaps BETWEEN utterances keep their true <sil> (they are real
    # assistant quiet, the barge-in surface). Timing tokens and yields
    # already on a tick keep their slot (only SIL upgrades).
    for t in ticks:
        if t.speech_active and t.emission == SIL:
            t.emission = SPEAK

    return GridSequence(
        scenario_id=tl.scenario_id, scenario_class=tl.scenario_class,
        n_ticks=n_ticks, ticks=ticks,
        audio=audio or {}, meta={**tl.meta,
                                 "label_contract": "v2-per-tick-speech"})


def from_scenario_dir(sc_dir: Path) -> GridSequence:
    """Load runs/<batch>/audio/<scenario>/ (timeline + stems) -> grid."""
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.synthetic.timeline import Timeline

    tl = Timeline.load(sc_dir / "timeline.actual.json")
    audio = {}
    for ch in ("user", "assistant", "background"):
        p = sc_dir / f"stem_{ch}.wav"
        if p.exists():
            audio[ch] = str(p)
    return build_grid(tl, audio)
