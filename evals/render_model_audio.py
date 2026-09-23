"""Render a free-run's word events as LISTENABLE audio (gallery batch).

Plain words: the diagnostic prints numbers; this turns a run into sound.
For each scenario it takes the in-slot audio the model was listening to
(user + background only — the model's own voice is never in its ears),
speaks every word the model chose, at the exact tick the model chose it
(80 ms per tick), and writes the mix into the layout viz.listen serves:

  <out>/audio/manifest.jsonl            one row per scenario
  <out>/audio/<scenario_id>/mixed.wav   conversation + model speech

  python -m evals.render_model_audio --picks runs/picks-cap-rows \
      --batch runs/synth-big7/audio --out runs/model-audio-cap-rows

Then on the local Mac: rsync the out dir into local runs/ and open the
gallery (python -m viz.listen -> http://localhost:8765). TTS runs on
the cluster (pylibs overlay); repeated words are synthesized once.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
TICK_SAMPLES = 1280  # 80 ms at 16 kHz


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True,
                    help="directory of <scenario_id>.json from "
                         "freerun_diag --save-picks")
    ap.add_argument("--batch", required=True,
                    help="factory batch dir with train_manifest.jsonl")
    ap.add_argument("--out", required=True,
                    help="output batch dir (gallery layout)")
    ap.add_argument("--engine", default="minicpm")
    ap.add_argument("--voice", default=None,
                    help="reference voice for the model's words "
                         "(default: the bakeoff USER_VOICE anchor)")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.synthetic.render import load_engine
    from evals.bakeoff.minicpm_bakeoff import USER_VOICE, find_assets
    from layera.prepare import mix_in_slot

    engine = load_engine(args.engine)
    assets = find_assets()
    ref_wav = Path(args.voice) if args.voice else assets / USER_VOICE

    picks_dir = Path(args.picks)
    batch = Path(args.batch)
    rows = [json.loads(l) for l in open(batch / "train_manifest.jsonl")]
    by_sid = {r["scenario_id"]: r for r in rows}

    out_audio = Path(args.out) / "audio"
    manifest = []
    # one TTS call per unique word, reused across placements/scenarios;
    # a failed synthesis is cached as None (skipped everywhere) — this
    # is an audition tool, one bad word must not lose the batch
    said: dict[str, np.ndarray | None] = {}
    tmp = Path(tempfile.mkdtemp(prefix="render-model-audio-"))

    def synthesize(text: str) -> np.ndarray | None:
        """TTS one word -> float32 mono at SR, or None after a retry."""
        for attempt in (1, 2):
            wp = tmp / f"w{len(said)}-{attempt}.wav"
            try:
                engine.tts(text, str(ref_wav), str(wp))
                x, sr = sf.read(wp, dtype="float32")
            except Exception as exc:  # MiniCPM-o crashes inside its own
                # model code on rare words (2048-code cap off-by-one,
                # job 6827931) and leaves no file behind
                print(f"[render]   tts failed ({attempt}/2) for "
                      f"{text!r}: {exc}", flush=True)
                continue
            if x.ndim > 1:
                x = x.mean(axis=1)
            if sr != SR:  # engine renders at its own rate
                t = np.arange(0, len(x)) * (SR / sr)
                x = np.interp(t, np.arange(0, len(x)), x)
            return x.astype(np.float32)
        return None

    n_scen = 0
    for pj in sorted(picks_dir.glob("*.json")):
        ev = json.loads(pj.read_text())
        sid = ev["scenario_id"]
        if sid not in by_sid:
            print(f"[render] {sid}: not in manifest, skipped")
            continue
        pcm = mix_in_slot(batch / by_sid[sid]["dir"], ev["n_ticks"])
        for w in ev["words"]:
            text = w["text"].strip()
            if not text:
                continue
            if text not in said:
                said[text] = synthesize(text)
            clip = said[text]
            if clip is None:
                continue
            s = w["tick"] * TICK_SAMPLES
            e = min(s + len(clip), len(pcm))
            if e > s:
                pcm[s:e] += clip[:e - s]
        peak = float(np.max(np.abs(pcm))) or 1.0
        if peak > 0.95:
            pcm *= 0.95 / peak
        sc = out_audio / sid
        sc.mkdir(parents=True, exist_ok=True)
        sf.write(sc / "mixed.wav", pcm, SR, subtype="PCM_16")
        manifest.append({"scenario_id": sid, "dir": sid,
                         "n_words": len(ev["words"]),
                         "voices": {"model": "tts(" + ref_wav.name + ")"}})
        n_scen += 1
        print(f"[render] {sid}: {len(ev['words'])} model words placed, "
              f"{len(pcm) / SR:.1f}s", flush=True)

    (out_audio / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in manifest))
    print(f"[render] done: {n_scen} scenarios -> {args.out} "
          f"(open the gallery: python -m viz.listen)")


if __name__ == "__main__":
    main()
