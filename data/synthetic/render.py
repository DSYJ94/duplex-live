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
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 24000
GRID_MS = 80
SHORT_CACHE_WORDS = 6  # <= this many words: memoize renders per (voice, text)

# anchor voice key -> reference clip in the MiniCPM-o HF snapshot assets/
# (proven clones; the pool from data.synthetic.voices adds diversity)
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
    from transformers import AutoModel, AutoTokenizer

    model_id = "openbmb/MiniCPM-o-4_5"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # AutoModel (not AutoModelForCausalLM): matches runs/minicpm_smoke.py —
    # pylibs-minicpm has no accelerate for the CausalLM init path
    model = AutoModel.from_pretrained(
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


# ---------------------------------------------------------------- engines
# A renderer engine turns (text, voice-ref clip path) into a wav. All engines
# share the voice pool, timelines, mix-time policies and the filter — only
# the vocoder differs. ADR-0002: open engines only.

class MiniCPMEngine:
    """openbmb/MiniCPM-o-4_5 zero-shot voice-cloned TTS (proven v0-v0.2)."""

    def __init__(self):
        self.model, _ = load_model()

    def tts(self, text: str, ref_wav: Path, out_path: Path) -> None:
        import librosa
        ref, _ = librosa.load(str(ref_wav), sr=16000, mono=True)
        tts_utterance(self.model, text, ref, out_path)


class ChatterboxEngine:
    """resemble-ai/chatterbox (Apache-2.0) zero-shot cloning with prosody
    controls — the engine class the PersonaPlex recipe was validated with.
    Runs under the pylibs-chatterbox overlay."""

    def __init__(self):
        from chatterbox.tts import ChatterboxTTS

        self.model = ChatterboxTTS.from_pretrained(device="cuda")

    def tts(self, text: str, ref_wav: Path, out_path: Path) -> None:
        import soundfile as sf
        import torch

        wav = self.model.generate(text, audio_prompt_path=str(ref_wav),
                                  exaggeration=0.4, cfg_weight=0.3,
                                  temperature=0.6)
        if isinstance(wav, torch.Tensor):
            wav = wav.squeeze().cpu().numpy()
        sf.write(str(out_path), wav, self.model.sr)


def load_engine(name: str):
    if name == "minicpm":
        return MiniCPMEngine()
    if name == "chatterbox":
        return ChatterboxEngine()
    raise ValueError(f"unknown engine {name!r} (minicpm | chatterbox)")


def load_voice_pool(voices_json: Path | None, assets: Path) -> dict[str, Path]:
    """voice key -> 16 kHz reference clip. Anchors are the 3 proven MiniCPM-o
    asset clones; voices.json (from data.synthetic.voices) adds the
    LibriSpeech CC-BY-4.0 pool. Missing/absent json -> anchors only."""
    pool = {k: assets / rel for k, rel in VOICES.items()}
    if voices_json is not None and Path(voices_json).exists():
        for e in json.loads(Path(voices_json).read_text()):
            pool[e["key"]] = Path(e["path"])
    return pool


def load_registers(voices_json: Path | None) -> dict[str, str]:
    """key -> low/mid/high F0 band, from voices_f0.json (pyin-measured,
    staged by the one-off measurement script next to voices.json)."""
    if voices_json is None:
        return {}
    f0p = Path(voices_json).parent / "voices_f0.json"
    if not f0p.exists():
        return {}
    d = json.loads(f0p.read_text())
    reg = {e["key"]: e["register"] for e in d.get("anchors", [])}
    reg.update({e["key"]: e.get("register") for e in d.get("pool", [])
                if e.get("register")})
    return reg


def assign_voices(tl, pool: dict[str, Path], anchor_prob: float,
                  registers: dict[str, str] | None = None) -> dict[str, str]:
    """One distinct voice per channel, deterministic in scenario_id, so
    timelines stay voice-agnostic and renders are reproducible. The user
    and assistant voices must differ in F0 register when measurable —
    same-register pairs are hard to attribute when auditioning the mix."""
    registers = registers or {}
    rng = random.Random(f"voices:{tl.scenario_id}")
    chs = sorted({e.ch for e in tl.events if e.text and e.kind != "marker"})
    picks: dict[str, str] = {}
    for ch in chs:
        anchor = CH_DEFAULT_VOICE.get(ch, "user")
        cand = [k for k in pool if k not in picks.values()]
        other = {"user": "assistant", "assistant": "user"}.get(ch)
        if other and other in picks and registers.get(picks[other]):
            diff_band = [k for k in cand
                         if registers.get(k) != registers.get(picks[other])]
            if diff_band:
                cand = diff_band
        if anchor in cand and rng.random() < anchor_prob:
            picks[ch] = anchor
        else:
            non_anchor = [k for k in cand if k != anchor] or cand
            picks[ch] = rng.choice(non_anchor)
    return picks


def cache_key(voice_key: str, text: str, engine: str = "") -> str:
    return f"{engine + '__' if engine else ''}{voice_key}__{hashlib.sha1(f'{voice_key}|{engine}|{text}'.encode()).hexdigest()[:16]}.wav"


def render_timeline(engine, tl, out_dir: Path, assets: Path, manifest,
                    pool: dict[str, Path] | None = None,
                    anchor_prob: float = 0.5,
                    registers: dict[str, str] | None = None,
                    engine_name: str = "minicpm",
                    clip_cache: Path | None = None) -> None:
    sc_dir = out_dir / tl.scenario_id
    utt_dir = sc_dir / "utterances"
    utt_dir.mkdir(parents=True, exist_ok=True)

    if pool is None:
        pool = {k: assets / rel for k, rel in VOICES.items()}
    channel_voice = assign_voices(tl, pool, anchor_prob, registers)
    tl.meta["voices"] = channel_voice

    def ref_key(ev) -> str:
        return ev.voice or channel_voice.get(
            ev.ch, CH_DEFAULT_VOICE.get(ev.ch, "user"))

    # 1. render every speech event. Short refs (<=6 words: backchannels,
    #    callouts) are babble-prone dice rolls — memoize each (voice, text)
    #    render in the clip cache. Bad clips get purged by the filter
    #    (purge_cache.py) and re-roll on the next render; passing clips
    #    freeze. Mix-time cuts still apply to the copied wavs as usual.
    speech = [e for e in tl.events if e.text and e.kind != "marker"]
    cache_hits = 0
    for i, ev in enumerate(speech):
        utt = utt_dir / f"{i:02d}_{ev.ch}.wav"
        vk = ref_key(ev)
        cached = (clip_cache / cache_key(vk, ev.text, engine_name)
                  if clip_cache is not None
                  and len(ev.text.split()) <= SHORT_CACHE_WORDS else None)
        if cached is not None and cached.exists():
            shutil.copyfile(cached, utt)
            cache_hits += 1
        else:
            engine.tts(ev.text, pool[vk], utt)
            if cached is not None:
                shutil.copyfile(utt, cached)
    if clip_cache is not None and speech:
        print(f"  [{tl.scenario_id}] clip-cache hits {cache_hits}/{len(speech)}",
              flush=True)
        audio, sr = sf.read(utt, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur_ms = round(len(audio) / sr * 1000)
        ev.actual_start_ms = ev.quantized_start()
        ev.actual_end_ms = ev.actual_start_ms + round(dur_ms / GRID_MS) * GRID_MS
        ev.wav = str(utt.relative_to(out_dir))
        print(f"  [{tl.scenario_id}] {ev.ch} @{ev.actual_start_ms}ms "
              f"{dur_ms}ms: {ev.text[:40]!r}", flush=True)

    # 2. retraction cut: any utterance overlapped by a barge_in on another
    #    channel is truncated at the barge instant (clean yield) with a short
    #    fade. MUST run before the repair below — the repair schedules
    #    same-channel events against post-cut ends (bug found by user
    #    audition 2026-09-17: yield pushed past the un-cut story end ->
    #    12-21 s reply latency after barge-ins)
    barges = [e for e in speech if e.kind == "barge_in"]
    for ev in speech:
        if ev.kind == "barge_in":
            continue
        cut_at = None
        for b in barges:
            if b.ch != ev.ch and ev.actual_start_ms < b.actual_start_ms < ev.actual_end_ms:
                cut_at = b.actual_start_ms if cut_at is None else min(cut_at, b.actual_start_ms)
        if cut_at is not None:
            audio, sr = sf.read(out_dir / ev.wav, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            keep = int((cut_at - ev.actual_start_ms) / 1000 * sr)
            fade = int(0.1 * sr)
            if keep > fade:
                audio[keep - fade:keep] *= np.linspace(1.0, 0.0, fade)
            sf.write(out_dir / ev.wav, audio[:keep], sr)
            ev.actual_end_ms = round(cut_at / GRID_MS) * GRID_MS
            ev.note = (ev.note + "; " if ev.note else "") + f"truncated at barge-in {cut_at}ms"
            print(f"  [{tl.scenario_id}] cut {ev.ch} utterance at {cut_at}ms (barge-in)", flush=True)

    # 2.5 barge-yield policy: assistant speech never talks over a barge-in —
    #     any other-channel utterance starting DURING a barge_in is pushed to
    #     the barge's actual end + 240 ms (human turn-taking offset). The
    #     generator can't know the realized barge duration; render time can.
    #     Trains clean-yield behaviour, never the talk-over-interrupter one.
    for b in barges:
        b_end = b.actual_end_ms
        for ev in speech:
            if ev.ch == b.ch or ev.kind == "barge_in":
                continue
            if b.actual_start_ms <= ev.actual_start_ms < b_end:
                shift = (b_end + 240) - ev.actual_start_ms
                ev.note = (ev.note + "; " if ev.note else "") + \
                    f"pushed {shift}ms (barge-yield policy)"
                print(f"  [{tl.scenario_id}] barge-yield: {ev.ch} event pushed "
                      f"{shift}ms behind barge end {b_end}ms", flush=True)
                ev.actual_start_ms += shift
                ev.actual_end_ms += shift

    # 3. same-channel repair: TTS can overrun the generator's duration
    #     estimate (measured 72–105 ms/char); push later same-channel events
    #     so a stem never sums two utterances. Cross-channel timing is never
    #     touched — overlap there is the point. Post-cut ends are what get
    #     mixed, so repairs are computed against them.
    by_ch: dict[str, list] = {}
    for ev in speech:
        by_ch.setdefault(ev.ch, []).append(ev)
    for ch, evs in by_ch.items():
        evs.sort(key=lambda e: e.actual_start_ms)
        for prev, cur in zip(evs, evs[1:]):
            min_start = prev.actual_end_ms + GRID_MS
            if cur.actual_start_ms < min_start:
                shift = min_start - cur.actual_start_ms
                cur.note = (cur.note + "; " if cur.note else "") + \
                    f"pushed {shift}ms (overrun repair)"
                print(f"  [{tl.scenario_id}] repair: {cur.ch} event pushed "
                      f"{shift}ms (prev overran estimate)", flush=True)
                cur.actual_start_ms = min_start
                cur.actual_end_ms += shift

    # 3. mix
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
        "voices": channel_voice,
        "engine": engine_name,
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
    ap.add_argument("--voices", default=None,
                    help="voices.json from data.synthetic.voices (pool mode)")
    ap.add_argument("--anchor-prob", type=float, default=0.5,
                    help="probability a channel keeps its proven anchor voice")
    ap.add_argument("--engine", default="minicpm",
                    choices=["minicpm", "chatterbox"],
                    help="TTS engine (see load_engine; overlays differ)")
    ap.add_argument("--clip-cache", default=None,
                    help="memoize <=6-word renders per (voice, text); purge "
                         "bad entries with purge_cache.py after filtering")
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

    pool = load_voice_pool(Path(args.voices) if args.voices else None, assets)
    registers = load_registers(Path(args.voices) if args.voices else None)
    print(f"voice pool: {len(pool)} voices"
          + (" (anchors only)" if len(pool) == len(VOICES) else "")
          + (f", {len(registers)} with F0 register" if registers else ""),
          flush=True)

    engine = load_engine(args.engine)
    clip_cache = Path(args.clip_cache) if args.clip_cache else None
    if clip_cache is not None:
        clip_cache.mkdir(parents=True, exist_ok=True)
        n = len(list(clip_cache.glob("*.wav")))
        print(f"clip-cache: {clip_cache} ({n} validated clips)", flush=True)
    with open(out / "manifest.jsonl", "a") as mf:
        for f in files:
            tl = Timeline.load(f)
            print(f"[render] {tl.scenario_id} ({tl.scenario_class})", flush=True)
            render_timeline(engine, tl, out, assets, mf,
                            pool=pool, anchor_prob=args.anchor_prob,
                            registers=registers, engine_name=args.engine,
                            clip_cache=clip_cache)
    print("done")


if __name__ == "__main__":
    main()
