"""Render UNSEEN-wording T-protocol prompts (2026-09-22).

The gate prompts in runs/bakeoff/prompts are already MiniCPM-TTS voice
(envelope check: CV 0.367), NOT tones — the sweep's "voice lessons vs
tone gate" split was wrong. The real split is SEEN script vs UNSEEN
wording: held-out factory lessons reuse trained utterances (Codex F3),
and v11 parrots those while muting on the gate's questions. This
renders the same T-protocol shapes with wording that appears nowhere
in the factory phrase bank (verified by grep), so the gate measures
generalisation instead of script recall. Same user voice as Phase A.

  python -m evals.render_unseen_prompts --out runs/bakeoff/prompts-unseen
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

UNSEEN = {
    # KEYS ARE THE PROTOCOL STEM NAMES — run.py's LIVE_TESTS loads
    # prompts_dir/<stem>.wav; the WORDING is what's unseen. First
    # render used u* keys and the gate crashed on missing t*.wav
    # (job 6781379); the cluster copies were renamed, not re-rendered.
    # T1: ten unseen factual questions (15 s cadence, like t1)
    **{f"t1_q{i}": q for i, q in enumerate([
        "What is twelve plus nine?",
        "How many days are in a leap year?",
        "What planet is closest to the sun?",
        "Who painted the Mona Lisa?",
        "What is the capital of Japan?",
        "How many minutes are in two hours?",
        "What do bees make?",
        "What is the largest ocean on Earth?",
        "How many sides does a hexagon have?",
        "What language is spoken in Brazil?",
    ], start=1)},
    # T2: long-answer bait + interruption, unseen topic
    "t2_bait": "Give me a long, detailed overview of how coffee is "
               "grown and harvested, from the plant to the cup.",
    "t2_interrupt": "Hold on — skip ahead. Just tell me which country "
                    "exports the most coffee, one sentence.",
    # T3: backchannels (model must NOT yield)
    "t3_bc1": "Right, right.",
    "t3_bc2": "Mm, okay.",
    "t3_bc3": "Yeah, go on.",
    # T4: two halves of one request, unseen topic
    "t4_part1": "Could you tell me, um, what exactly...",
    "t4_part2": "...the difference between a virus and a bacterium "
                "is, in one sentence?",
    # T5: background speech, model never addressed
    "t5_bg1": "Did you catch the bus this morning? It was packed again.",
    "t5_bg2": "Yeah, and then it was late leaving the stop, like always.",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bakeoff/prompts-unseen")
    ap.add_argument("--engine", choices=["minicpm", "chatterbox"],
                    default="minicpm")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import librosa
    import numpy as np

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.synthetic.render import load_engine
    from evals.bakeoff.minicpm_bakeoff import USER_VOICE, find_assets

    engine = load_engine(args.engine)
    assets = find_assets()
    ref_wav = assets / USER_VOICE

    for stem, text in UNSEEN.items():
        wav_path = out_dir / f"{stem}.wav"
        if wav_path.exists():
            continue
        t0 = time.time()
        engine.tts(text, ref_wav, wav_path)
        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
        print(f"[unseen-tts] {stem}: {len(wav) / sr:.1f}s "
              f"({time.time() - t0:.0f}s) {text[:40]!r}", flush=True)
    print(f"[unseen-tts] done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
