"""Render grid-aligned Timelines to duplex audio via MiniCPM-o 4.5 TTS.

One model load, then per event: zero-shot voice-cloned TTS (per-channel voice),
mixed onto a 24 kHz canvas at the event's grid-quantized start. Writes:

  <out>/<scenario_id>/mixed.wav        all channels mixed (mono, 24 kHz)
  <out>/<scenario_id>/stem_<ch>.wav    per-channel stems
  <out>/<scenario_id>/utterances/*.wav individual event renders
  <out>/manifest.jsonl                 one line per scenario, actual timings

Runs on Isambard under the pylibs-minicpm overlay (transformers 4.51.0) —
see synth-render.slurm. ADR-0002: MiniCPM-o is Apache-2.0; outputs are legal
training data. PersonaPlex outputs must NEVER be used here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 24000
GRID_MS = 80

# voice key -> reference clip in the MiniCPM-o HF snapshot assets/
VOICES = {
    "assistant": "system_ref_audio.wav",
    "user": "HT_ref_audio.wav",
    "background": "system_ref_audio_2.wav",
}
CH_DEFAULT_VOICE = {"user": "user", "assistant": "assistant", "background": "background"}

READ_PROMPT = "请朗读以下内容。 "
SYS_PRE = "模仿音频样本的音色并生成新的内容。"
SYS_POST = "请用这种声音风格来为用户提供帮助。 直接作答，不要有冗余内容"


def snapshot_assets() -> Path:
    root = Path(__file__).resolve().parents[2]
    snaps = list(root.glob("models/hf-cache/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets"))
    if not snaps:  # repo checkout: assets live on Isambard, pass --assets
        raise FileNotFoundError("MiniCPM-o snapshot assets not found; pass --assets")
    return snaps[0]


def load_model():
    import librosa  # noqa: F401  (model remote code expects it)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "openbmb/MiniCPM-o-4_5"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, init_tts=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    model.init_tts()
    return model, tok


def tts_utterance(model, text: str, ref_audio: np.ndarray, out_path: Path) -> None:
    import librosa

    model.reset_session()
    msgs = [
        {"role": "system", "content": [SYS_PRE, ref_audio, SYS_POST]},
        {"role": "user", "content": [READ_PROMPT + text]},
    ]
    model.chat(msgs=msgs, do_sample=False, max_new_tokens=2048,
               use_tts_template=True, generate_audio=True, temperature=0.1,
               output_audio_path=str(out_path))


def render_timeline(model, tl, out_dir: Path, assets: Path, manifest) -> None:
    import librosa

    sc_dir = out_dir / tl.scenario_id
    utt_dir = sc_dir / "utterances"
    utt_dir.mkdir(parents=True, exist_ok=True)

    refs: dict[str, np.ndarray] = {}

    def ref_for(ev) -> np.ndarray:
        key = ev.voice or CH_DEFAULT_VOICE.get(ev.ch, "user")
        if key not in refs:
            wav, _ = librosa.load(assets / VOICES[key], sr=16000, mono=True)
            refs[key] = wav
        return refs[key]

    # 1. render every speech event
    speech = [e for e in tl.events if e.text and e.kind != "marker"]
    for i, ev in enumerate(speech):
        utt = utt_dir / f"{i:02d}_{ev.ch}.wav"
        tts_utterance(model, ev.text, ref_for(ev), utt)
        audio, sr = sf.read(utt, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur_ms = round(len(audio) / sr * 1000)
        ev.actual_start_ms = ev.quantized_start()
        ev.actual_end_ms = ev.actual_start_ms + round(dur_ms / GRID_MS) * GRID_MS
        ev.wav = str(utt.relative_to(out_dir))
        print(f"  [{tl.scenario_id}] {ev.ch} @{ev.actual_start_ms}ms "
              f"{dur_ms}ms: {ev.text[:40]!r}", flush=True)

    # 2. mix
    total_ms = tl.duration_ms() + 1000
    n = int(total_ms / 1000 * SR)
    channels = sorted({e.ch for e in speech})
    stems = {ch: np.zeros(n, dtype=np.float32) for ch in channels}
    for ev in speech:
        audio, sr = sf.read(out_dir / ev.wav, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
        off = int(ev.actual_start_ms / 1000 * SR)
        end = min(off + len(audio), n)
        stems[ev.ch][off:end] += audio[: end - off] * ev.gain

    mixed = np.sum([stems[ch] for ch in channels], axis=0)
    peak = float(np.max(np.abs(mixed))) or 1.0
    if peak > 0.95:
        mixed *= 0.95 / peak
    sf.write(sc_dir / "mixed.wav", mixed, SR)
    for ch, st in stems.items():
        sf.write(sc_dir / f"stem_{ch}.wav", st, SR)

    tl.save(sc_dir / "timeline.actual.json")
    manifest.write(json.dumps({
        "scenario_id": tl.scenario_id,
        "scenario_class": tl.scenario_class,
        "dir": str(sc_dir.relative_to(out_dir)),
        "mixed_wav": f"{tl.scenario_id}/mixed.wav",
        "sr": SR,
        "grid_ms": GRID_MS,
        "duration_ms": total_ms,
        **tl.overlap_stats(),
        "tts_engine": "openbmb/MiniCPM-o-4_5 (Apache-2.0) zero-shot voice-cloned TTS",
        "license": "synthetic; open models only (ADR-0002); no closed-API content",
    }, ensure_ascii=False) + "\n")
    manifest.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timelines", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--only", default=None, help="scenario_id substring filter")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from data.synthetic.timeline import Timeline

    assets = Path(args.assets) if args.assets else snapshot_assets()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.timelines).glob("*.json"))
    if args.only:
        files = [f for f in files if args.only in f.name]
    print(f"{len(files)} timelines to render")

    model, _ = load_model()
    with open(out / "manifest.jsonl", "a") as mf:
        for f in files:
            tl = Timeline.load(f)
            print(f"[render] {tl.scenario_id} ({tl.scenario_class})", flush=True)
            render_timeline(model, tl, out, assets, mf)
    print("done")


if __name__ == "__main__":
    main()
