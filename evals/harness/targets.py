"""Live targets for the eval harness (Issue #6).

A Target is a duplex system under test driven chunk-by-chunk. The replay
path scores recordings; live targets CREATE recordings in the same layout
(<test>/{user_in.wav, model_out.wav, log.jsonl}) so one metrics code path
scores both. Chunk telemetry includes the frameHash diagnostic (sha16 of
each stimulus chunk — the qwen-live traceability pattern).

MiniCPMDuplexTarget ports the proven bake-off driver (Phase B of
evals/bakeoff/minicpm_bakeoff.py) behind the Target seam. LayerATarget
arrives with Issue #7: same seam, tick loop instead of 1 s chunks.
"""

from __future__ import annotations

import abc
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evals.harness.metrics import chunk_hash  # noqa: E402

SR_IN = 16000
SR_OUT = 24000


class Target(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def prepare(self, **kw) -> None: ...

    @abc.abstractmethod
    def feed(self, chunk: np.ndarray) -> dict: ...

    def finish(self) -> dict:  # optional teardown
        return {}


class MiniCPMDuplexTarget(Target):
    """MiniCPM-o 4.5 duplex session (1 s chunks), as in the bake-off."""

    name = "minicpm-duplex"

    def __init__(self):
        self.model = None
        self.duplex = None
        self.assets = None

    def prepare(self, assets: Path, voice: str = "HT_ref_audio.wav") -> None:
        import librosa
        import torch
        from transformers import AutoModel

        self.assets = Path(assets)
        ref_audio, _ = librosa.load(self.assets / voice, sr=16000, mono=True)
        self.model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-o-4_5", trust_remote_code=True,
            attn_implementation="sdpa", torch_dtype=torch.bfloat16,
            init_vision=True, init_audio=True, init_tts=True).eval().cuda()
        self.model.init_tts()
        self.duplex = self.model.as_duplex()
        self.duplex.prepare(
            prefix_system_prompt="Streaming Omni Conversation. "
            "Always reply in English only.",
            ref_audio=ref_audio,
            prompt_wav_path=str(self.assets / voice))

    def feed(self, chunk: np.ndarray) -> dict:
        import time

        t0 = time.time()
        pre = self.duplex.streaming_prefill(audio_waveform=chunk)
        res = self.duplex.streaming_generate(
            prompt_wav_path=str(self.assets / "HT_ref_audio.wav"))
        wall = time.time() - t0
        wav = res.get("audio_waveform")
        if wav is not None:
            wav = np.asarray(wav if isinstance(wav, np.ndarray)
                             else wav.cpu().numpy(), dtype=np.float32).reshape(-1)
        else:
            wav = np.zeros(0, dtype=np.float32)
        return {"wall_s": round(wall, 3), "prefill_ok": pre.get("success"),
                "is_listen": res.get("is_listen"),
                "text": res.get("text", ""),
                "end_of_turn": res.get("end_of_turn"),
                "cost_llm": round(res.get("cost_llm", 0.0), 3),
                "cost_tts": round(res.get("cost_tts", 0.0), 3),
                "cost_token2wav": round(res.get("cost_token2wav", 0.0), 3),
                "out_samples": int(len(wav)),
                "chunk_sha16": chunk_hash(chunk),
                "_wav": wav}


def _tone_80ms() -> np.ndarray:
    """Quiet 220 Hz sine, 4 ms raised-cosine edges, at the 24 kHz out
    rate — a deliberately synthetic stand-in for the M2 voice, so the
    energy metrics measure the model's speaking DECISIONS."""
    sr = 24000
    n = int(sr * 0.08)
    t = np.arange(n) / sr
    w = (0.06 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    edge = int(sr * 0.004)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge)).astype(np.float32)
    env = np.ones(n, np.float32)
    env[:edge] = ramp
    env[-edge:] = ramp[::-1]
    return w * env


class LayerATarget(Target):
    """Phase-1 duplex thinker behind the Target seam (Issue #7).

    Each 1 s chunk is cut into 80 ms ticks. Every tick: user audio enters
    the KV (append-only), then the TRAINED model decides the beat —
    <sil> stays quiet, word pieces speak (tone), <yield>/<t+Ns> are
    logged and stay silent. Countdown/timing behavior lives in the log
    (n_timing per chunk) until the T7 timed metrics land."""

    name = "layera-p1"

    def __init__(self):
        self.thinker = None
        self._buf = np.zeros(0, dtype=np.float32)
        self._speaking = False
        self._tone = None

    def prepare(self, ckpt, **kw) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from layera.duplex import DuplexThinker, TICK_SAMPLES
        self._tick_n = TICK_SAMPLES
        self._tone = _tone_80ms()
        self.thinker = DuplexThinker(str(ckpt))

    def feed(self, chunk: np.ndarray) -> dict:
        import time

        t0 = time.time()
        self._buf = np.concatenate([self._buf,
                                    chunk.astype(np.float32)])
        texts = []
        n_yield = n_timing = n_control = 0
        wav_out = []
        end_of_turn = False
        ticks = 0
        while len(self._buf) >= self._tick_n:
            tick_pcm = self._buf[:self._tick_n]
            self._buf = self._buf[self._tick_n:]
            self.thinker.feed_tick(tick_pcm)
            tok, _s = self.thinker.decide()
            ticks += 1
            kind = self.thinker.kind_of(tok)
            if kind == "word":
                self._speaking = True
                texts.append(self.thinker.proc.tokenizer.decode([tok]))
                wav_out.append(self._tone)
            elif kind == "sil":
                if self._speaking:
                    end_of_turn = True
                self._speaking = False
            elif kind == "yield":
                n_yield += 1
                self._speaking = False
            elif kind == "timing":
                n_timing += 1
                self._speaking = False
            else:  # <delegate>/<barge-in> or unexpected
                n_control += 1
                self._speaking = False
        wav = np.concatenate(wav_out) if wav_out else np.zeros(0, np.float32)
        return {"wall_s": round(time.time() - t0, 3), "ticks": ticks,
                "text": "".join(texts).strip(),
                "is_listen": not self._speaking,
                "end_of_turn": end_of_turn,
                "n_yield": n_yield, "n_timing": n_timing,
                "n_control": n_control,
                "out_samples": int(len(wav)),
                "chunk_sha16": chunk_hash(chunk),
                "_wav": wav}


def run_live(target: Target, tests: dict, out_root: Path,
             prompts_dir: Path) -> None:
    """Drive a live target through the T-protocol; write replay-layout
    recordings. tests: {name: (spec[(start_s, stem)], total_s)}."""
    import librosa

    for name, (spec, total_s) in tests.items():
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        timeline = np.zeros(total_s * SR_IN, dtype=np.float32)
        for start_s, stem in spec:
            wav, _ = librosa.load(prompts_dir / f"{stem}.wav", sr=SR_IN, mono=True)
            i0 = start_s * SR_IN
            i1 = min(i0 + len(wav), len(timeline))
            timeline[i0:i1] += wav[: i1 - i0]
        timeline = np.clip(timeline, -1.0, 1.0)
        sf.write(out_dir / "user_in.wav", timeline, SR_IN)

        out_audio, rows = [], []
        with open(out_dir / "log.jsonl", "w") as logf:
            for c in range(int(np.ceil(len(timeline) / SR_IN))):
                chunk = timeline[c * SR_IN:(c + 1) * SR_IN]
                if len(chunk) < SR_IN:
                    chunk = np.concatenate(
                        [chunk, np.zeros(SR_IN - len(chunk), np.float32)])
                rec = target.feed(chunk)
                wav = rec.pop("_wav")
                rec["chunk"], rec["t_s"] = c, c
                rows.append(rec)
                logf.write(json.dumps(rec) + "\n")
                if len(wav):
                    out_audio.append(wav)
                if c % 20 == 0 or rec["text"].strip():
                    print(f"[{name}] chunk {c:4d} listen={rec['is_listen']} "
                          f"text={rec['text'][:50]!r}", flush=True)
        if out_audio:
            sf.write(out_dir / "model_out.wav", np.concatenate(out_audio), SR_OUT)
        print(f"[{name}] live run done -> {out_dir}", flush=True)
