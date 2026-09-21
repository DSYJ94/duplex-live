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

    def reset(self) -> None:  # fresh session between tests
        return None


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

    v2: the thinker sees audio in the EXACT training format (no chat
    envelope; ear features over 4 s windows; one decision per feature —
    see layera/duplex.py and the freerun diagnostic in EXPERIMENTS.md).
    Beats arrive with their audio timestamp, so speaking decisions are
    placed on an output tape at the right wall-clock offset regardless
    of the windowed feature lag; render() assembles model_out.wav.
    Word beats speak (tone), <sil> ends turns, <yield>/<t+Ns> are
    logged and stay silent."""

    name = "layera-p1"

    def __init__(self):
        self.thinker = None
        self._speaking = False
        self._tone = None
        self._placed = []  # (out-sample offset) of each spoken beat

    def prepare(self, ckpt, **kw) -> None:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from layera.duplex import DuplexThinker
        self._tone = _tone_80ms()
        self.thinker = DuplexThinker(str(ckpt))

    def _apply(self, beats) -> dict:
        texts, n_yield, n_timing, n_control = [], 0, 0, 0
        end_of_turn = False
        out_samples = 0
        for tok, t_s in beats:
            kind = self.thinker.kind_of(tok)
            if kind == "word":
                self._speaking = True
                texts.append(self.thinker.proc.tokenizer.decode([tok]))
                start = int(t_s * SR_OUT)
                self._placed.append(start)
                out_samples += len(self._tone)
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
        return {"text": "".join(texts).strip(),
                "end_of_turn": end_of_turn,
                "n_yield": n_yield, "n_timing": n_timing,
                "n_control": n_control, "out_samples": out_samples}

    def feed(self, chunk: np.ndarray) -> dict:
        import time

        t0 = time.time()
        beats = self.thinker.feed_audio(chunk)
        rec = self._apply(beats)
        rec.update({"wall_s": round(time.time() - t0, 3),
                    "ticks": len(beats),
                    "is_listen": not self._speaking,
                    "chunk_sha16": chunk_hash(chunk),
                    "_wav": np.zeros(0, np.float32)})
        return rec

    def finish(self) -> dict:
        beats = self.thinker.finish()
        rec = self._apply(beats)
        rec.update({"ticks": len(beats),
                    "is_listen": not self._speaking})
        return rec

    def reset(self) -> None:
        self.thinker.reset()
        self._placed = []
        self._speaking = False

    def render(self, total_s: float) -> np.ndarray:
        """Assemble model_out.wav from the placed speaking beats. One
        tone-length of headroom: the last beats can sit a step past the
        timeline edge (window tails)."""
        n = int(np.ceil(total_s * SR_OUT))
        tape = np.zeros(n + len(self._tone), dtype=np.float32)
        for start in self._placed:
            tape[start:start + len(self._tone)] += self._tone
        return tape[:n]


def run_live(target: Target, tests: dict, out_root: Path,
             prompts_dir: Path) -> None:
    """Drive a live target through the T-protocol; write replay-layout
    recordings. tests: {name: (spec[(start_s, stem)], total_s)}."""
    import librosa

    for name, (spec, total_s) in tests.items():
        if hasattr(target, "reset"):
            target.reset()  # fresh session per test — no cross-talk
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
                wav = rec.pop("_wav", None)
                rec["chunk"], rec["t_s"] = c, c
                rows.append(rec)
                logf.write(json.dumps(rec) + "\n")
                if wav is not None and len(wav):
                    out_audio.append(wav)
                if c % 20 == 0 or rec["text"].strip():
                    print(f"[{name}] chunk {c:4d} listen={rec['is_listen']} "
                          f"text={rec['text'][:50]!r}", flush=True)
            fin = target.finish() if hasattr(target, "finish") else {}
            if fin:
                fin["chunk"], fin["t_s"] = len(rows), len(rows)
                rows.append(fin)
                logf.write(json.dumps(fin) + "\n")
        total_s = len(timeline) / SR_IN
        if hasattr(target, "render"):
            wav_out = target.render(total_s)
            if len(wav_out):
                sf.write(out_dir / "model_out.wav", wav_out, SR_OUT)
        elif out_audio:
            sf.write(out_dir / "model_out.wav", np.concatenate(out_audio), SR_OUT)
        print(f"[{name}] live run done -> {out_dir}", flush=True)
